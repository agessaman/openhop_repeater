"""The per-message channel flood scope is Core's; the repeater must only inherit it.

openHop Core owns the feature end to end (``docs/openhop-frame-extensions.md``
there, and its own protocol tests). What Core cannot check is that *this*
package's subclasses still reach it: ``RepeaterCompanionBridge`` and
``CompanionFrameServer`` both sit between a client and Core's send path, and
either could shadow a method or drop a keyword without any Core test noticing.

So these are wiring assertions, deliberately thin. Protocol behaviour --
byte layouts, error mapping, key validation -- belongs in Core and is not
duplicated here.

The second test is the one that matters on a repeater specifically: the
dispatcher re-resolves flood scope for everything it sends, so a scoped
companion packet only survives because Core marks it decided. If that mark
were lost, the message would go out under the repeater's own region instead of
the one the client asked for, and nothing would report an error.
"""

import struct
from unittest.mock import AsyncMock, Mock

import pytest

from openhop_core import LocalIdentity
from openhop_core.companion.constants import (
    CMD_SEND_CHANNEL_TXT_MSG,
    OPENHOP_CHANNEL_SCOPE_PROBE,
    OPENHOP_CHANNEL_TXT_SCOPED,
    OPENHOP_EXTENSION_MARKER,
    OPENHOP_SCOPE_PROBE_RESERVED_LEN,
    RESP_CODE_OK,
    RESP_CODE_OPENHOP_EXTENSION,
)
from openhop_core.companion.models import Channel
from openhop_core.protocol.constants import ROUTE_TYPE_TRANSPORT_FLOOD
from openhop_core.protocol.transport_keys import calc_transport_code, get_auto_key_for
from repeater.companion.bridge import RepeaterCompanionBridge
from repeater.companion.frame_server import CompanionFrameServer

SCOPE_KEY = get_auto_key_for("#USA")


def _frame_server(bridge):
    """A repeater frame server whose outbound frames land in a list."""
    server = CompanionFrameServer(bridge, "0x77", port=0)
    frames: list[bytes] = []
    server._write_frame = frames.append
    return server, frames


@pytest.mark.asyncio
async def test_frame_server_subclass_inherits_core_extension_handlers():
    """The repeater's CompanionFrameServer must not shadow command 3."""
    bridge = Mock()
    bridge.get_public_key = Mock(return_value=bytes(range(32)))
    bridge.get_channel = Mock(return_value=Channel(name="general", secret=bytes(16)))
    bridge.send_channel_message = AsyncMock(return_value=True)
    server, frames = _frame_server(bridge)

    probe = bytes([CMD_SEND_CHANNEL_TXT_MSG, OPENHOP_CHANNEL_SCOPE_PROBE])
    probe += bytes(OPENHOP_SCOPE_PROBE_RESERVED_LEN)
    await server._handle_cmd(probe)

    send = bytes([CMD_SEND_CHANNEL_TXT_MSG, OPENHOP_CHANNEL_TXT_SCOPED, 1])
    send += struct.pack("<I", 1234) + SCOPE_KEY + b"hello"
    await server._handle_cmd(send)

    assert frames[0][0] == RESP_CODE_OPENHOP_EXTENSION
    assert frames[0][1:7] == OPENHOP_EXTENSION_MARKER
    assert frames[1] == bytes([RESP_CODE_OK])
    bridge.send_channel_message.assert_awaited_once_with(
        1, "hello", timestamp=1234, flood_scope_key=SCOPE_KEY
    )


@pytest.mark.asyncio
async def test_scoped_send_through_repeater_bridge_reaches_injector_scoped():
    """End to end through RepeaterCompanionBridge, with the dispatcher's mark set."""
    injected = []

    async def injector(pkt, **kwargs):
        injected.append(pkt)
        return True

    bridge = RepeaterCompanionBridge(LocalIdentity(), injector, node_name="rep")
    bridge.channels.set(0, Channel(name="test-ch", secret=b"\xab" * 16))

    assert await bridge.send_channel_message(0, "hello", flood_scope_key=SCOPE_KEY) is True

    assert len(injected) == 1
    pkt = injected[0]
    assert pkt.get_route_type() == ROUTE_TYPE_TRANSPORT_FLOOD
    assert pkt.transport_codes[0] == calc_transport_code(SCOPE_KEY, pkt)
    assert pkt.transport_codes[1] == 0
    # Without this the shared dispatcher would re-scope the packet to the
    # repeater's own region on its way out.
    assert pkt._flood_scope_applied is True
