"""
Discovery request/response handling helper for pyMC Repeater.

This module handles the processing and response to discovery requests,
allowing other nodes to discover repeaters on the mesh network.
"""

import asyncio
import logging
import secrets

from pymc_core.node.handlers.anon_request import AnonRateLimiter
from pymc_core.node.handlers.control import ControlHandler

logger = logging.getLogger("DiscoveryHelper")

# Default upper bound (ms) for the randomized pre-send jitter applied to node
# discovery responses. A node-discover request is a broadcast that every
# in-range repeater answers at once, so without jitter they all transmit at the
# same engine-scheduled instant and collide. Mirrors the firmware, which spreads
# these replies deliberately (MyMesh.cpp:797, sendZeroHop with
# getRetransmitDelay*4). Safe to be generous: the requester's discovery window is
# 60s (firmware pending_discover_until = futureMillis(60000)).
DEFAULT_DISCOVERY_RESPONSE_JITTER_MS = 2000


class DiscoveryHelper:
    """Helper class for processing discovery requests in the repeater."""

    def __init__(
        self,
        local_identity,
        packet_injector=None,
        node_type: int = 2,
        log_fn=None,
        debug_log_fn=None,
        response_jitter_ms: int = DEFAULT_DISCOVERY_RESPONSE_JITTER_MS,
        forwarding_enabled_fn=None,
        mod_timestamp_fn=None,
        rate_limit_max: int = 0,
        rate_limit_secs: float = 120.0,
    ):
        """
        Initialize the discovery helper.

        Args:
            local_identity: The LocalIdentity instance for this repeater
            packet_injector: Callable to inject new packets into the router for sending
            node_type: Node type identifier (2 = Repeater)
            log_fn: Optional logging function for ControlHandler
            debug_log_fn: Optional logging for verbose ControlHandler messages (e.g. callback
                presence). Pass logger.debug to avoid INFO noise when forwarding to companions.
            response_jitter_ms: Upper bound (ms) for the randomized delay added before
                transmitting a discovery response, to avoid multiple repeaters colliding
                when answering the same broadcast. Set to 0 to disable (e.g. in tests).
            forwarding_enabled_fn: Optional ``() -> bool`` returning whether this repeater is
                currently forwarding. Mirrors firmware ``!_prefs.disable_fwd``: a non-forwarding
                repeater (monitor / no_tx mode) does not answer discovery requests. Defaults to
                always-enabled.
            mod_timestamp_fn: Optional ``() -> int`` returning the unix timestamp of the last
                discovery-relevant change for this node. Mirrors firmware
                ``_prefs.discovery_mod_timestamp``: only reply when ``mod_timestamp >= since``.
                When ``None`` the ``since`` filter is ignored and we always reply (the official
                client sends ``since == 0`` anyway).
            rate_limit_max: Max discovery replies per ``rate_limit_secs`` window. Firmware uses
                ``discover_limiter(4, 120)``. **Disabled by default (0)** so the repeater answers
                every discovery, matching this app's long-standing behaviour; rapid manual retries
                would otherwise be silently dropped. Set > 0 to opt into firmware-style limiting.
            rate_limit_secs: Window length (seconds) for ``rate_limit_max``. Firmware uses 120.
        """
        self.local_identity = local_identity
        self.packet_injector = packet_injector  # Function to inject packets into router
        self.node_type = node_type
        self.response_jitter_ms = max(0, int(response_jitter_ms))
        self.forwarding_enabled_fn = forwarding_enabled_fn
        self.mod_timestamp_fn = mod_timestamp_fn

        # Optional rate limiter so a burst of discovery broadcasts can't make us a flood
        # amplifier (firmware ``discover_limiter(4, 120)``). Off by default: enabling it can
        # silently drop responses to legitimate repeated discovery, which surprised users.
        self.discover_limiter = (
            AnonRateLimiter(maximum=int(rate_limit_max), secs=float(rate_limit_secs))
            if rate_limit_max and int(rate_limit_max) > 0
            else None
        )

        # Create ControlHandler internally as a parsing utility
        self.control_handler = ControlHandler(
            log_fn=log_fn or logger.info,
            debug_log_fn=debug_log_fn,
        )
        self._pending_tasks = set()

        # Set up the request callback
        self.control_handler.set_request_callback(self._on_discovery_request)
        logger.debug("Discovery handler initialized")

    def _track_task(self, task: asyncio.Task) -> None:
        self._pending_tasks.add(task)

        def _on_done(done_task: asyncio.Task) -> None:
            self._pending_tasks.discard(done_task)
            try:
                done_task.result()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Background discovery task failed: {e}", exc_info=True)

        task.add_done_callback(_on_done)

    def _on_discovery_request(self, request_data: dict) -> None:
        """
        Handle incoming discovery request.

        Args:
            request_data: Dictionary containing the parsed discovery request
        """
        try:
            tag = request_data.get("tag", 0)
            filter_byte = request_data.get("filter", 0)
            prefix_only = request_data.get("prefix_only", False)
            snr = request_data.get("snr", 0.0)
            rssi = request_data.get("rssi", 0)
            since = request_data.get("since", 0)

            logger.info(
                f"Request: tag=0x{tag:08X}, filter=0x{filter_byte:02X}, "
                f"SNR={snr:+.1f}dB, RSSI={rssi}dBm"
            )

            # Don't answer discovery while forwarding is disabled (monitor / no_tx mode).
            # Mirrors firmware ``!_prefs.disable_fwd`` guard in onControlDataRecv.
            if self.forwarding_enabled_fn is not None and not self.forwarding_enabled_fn():
                logger.debug("Forwarding disabled, not answering discovery")
                return

            # Check if filter matches our node type (repeater = 2, filter_mask = 0x04)
            filter_mask = 1 << self.node_type  # 1 << 2 = 0x04
            if (filter_byte & filter_mask) == 0:
                logger.debug("Filter doesn't match, ignoring")
                return

            # Honor the request's ``since`` filter: firmware only replies when its
            # discovery info changed at/after ``since`` (_prefs.discovery_mod_timestamp).
            # With no mod-timestamp source we always reply (official client sends since=0).
            if since and self.mod_timestamp_fn is not None:
                mod_ts = int(self.mod_timestamp_fn())
                if mod_ts < since:
                    logger.debug(
                        f"Discovery info unchanged since {since} (mod_ts={mod_ts}), ignoring"
                    )
                    return

            # Optional rate limit (firmware discover_limiter); off by default so repeated
            # discovery isn't silently dropped.
            if self.discover_limiter is not None and not self.discover_limiter.allow():
                logger.debug("Discovery rate limit reached, dropping response")
                return

            logger.info("Sending response...")

            if self.local_identity:
                self._send_discovery_response(tag, self.node_type, snr, prefix_only)
            else:
                logger.warning("No local identity available for response")

        except Exception as e:
            logger.error(f"Error handling request: {e}")

    def _send_discovery_response(
        self,
        tag: int,
        node_type: int,
        inbound_snr: float,
        prefix_only: bool,
    ) -> None:
        """
        Create and send a discovery response packet.

        Args:
            tag: The tag from the discovery request
            node_type: Node type identifier
            inbound_snr: SNR of the received request
            prefix_only: Whether to use prefix-only mode
        """
        try:
            our_pub_key = self.local_identity.get_public_key()

            from pymc_core.protocol.packet_builder import PacketBuilder

            response_packet = PacketBuilder.create_discovery_response(
                tag=tag,
                node_type=node_type,
                inbound_snr=inbound_snr,
                pub_key=our_pub_key,
                prefix_only=prefix_only,
            )

            # Send response via router injection
            if self.packet_injector:
                task = asyncio.create_task(self._send_packet_async(response_packet, tag))
                self._track_task(task)
            else:
                logger.warning("No packet injector available - discovery response not sent")

        except Exception as e:
            logger.error(f"Error creating discovery response: {e}")

    async def _send_packet_async(self, packet, tag: int) -> None:
        """
        Send a discovery response packet via router injection.

        Args:
            packet: The packet to send
            tag: The tag for logging purposes
        """
        try:
            # Randomized pre-send jitter so multiple repeaters answering the same
            # zero-hop discovery broadcast don't transmit at the same engine-scheduled
            # instant and collide (the engine's DIRECT delay is fixed, not random).
            # Mirrors firmware MyMesh.cpp:797. Uses secrets like the engine's TX jitter.
            if self.response_jitter_ms > 0:
                jitter_s = secrets.randbelow(self.response_jitter_ms + 1) / 1000.0
                if jitter_s > 0:
                    logger.debug(
                        f"Discovery response jitter {jitter_s * 1000:.0f}ms for tag 0x{tag:08X}"
                    )
                    await asyncio.sleep(jitter_s)

            success = await self.packet_injector(packet, wait_for_ack=False)
            if success:
                logger.info(f"Response sent for tag 0x{tag:08X}")
            else:
                logger.warning(f"Failed to send response for tag 0x{tag:08X}")
        except Exception as e:
            logger.error(f"Error sending response: {e}")
