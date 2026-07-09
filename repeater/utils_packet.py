import logging
from typing import Optional, Tuple

from openhop_core.protocol import PacketBuilder
from openhop_core.protocol.constants import ROUTE_TYPE_TRANSPORT_FLOOD

logger = logging.getLogger("RepeaterPacketUtils")


def create_scoped_advert_packet(
    *,
    local_identity,
    node_name: str,
    latitude: float,
    longitude: float,
    flags: int,
    default_region,
    scope_label: str,
    storage=None,
) -> Tuple[object, Optional[str]]:
    """Create a flood advert packet and apply default-region transport scope when configured."""
    packet = PacketBuilder.create_advert(
        local_identity=local_identity,
        name=node_name,
        lat=latitude,
        lon=longitude,
        feature1=0,
        feature2=0,
        flags=flags,
        route_type="flood",
    )

    scoped_region_name = _apply_default_region_scope(
        packet=packet,
        default_region=default_region,
        scope_label=scope_label,
        storage=storage,
    )
    return packet, scoped_region_name


def _apply_default_region_scope(
    *, packet, default_region, scope_label: str, storage=None
) -> Optional[str]:
    """Apply transport-flood scoping for a default region if provided."""
    region_name = str(default_region).strip() if default_region not in (None, "") else ""
    if not region_name:
        return None

    try:
        from openhop_core.protocol.transport_keys import calc_transport_code, get_auto_key_for

        if region_name.startswith("$"):
            # Private region: firmware loads provisioned keys from the
            # TransportKeyStore (loadKeysFor), never a name-derived auto key.
            # No provisioned key leaves default_scope null in firmware, which
            # falls back to a plain unscoped flood — mirror that here.
            region_key = _stored_region_key(storage, region_name)
            if region_key is None:
                logger.warning(
                    "No provisioned key for private default region '%s'; "
                    "sending %s as unscoped flood",
                    region_name,
                    scope_label,
                )
                return None
        else:
            region_key = get_auto_key_for(region_name)

        packet.transport_codes[0] = calc_transport_code(region_key, packet)
        packet.transport_codes[1] = 0  # reserved for home region
        packet.header = (packet.header & ~0x03) | ROUTE_TYPE_TRANSPORT_FLOOD
        return region_name
    except Exception as scope_err:
        logger.warning(
            "Failed to apply default region scope '%s' to %s; sending unscoped flood: %s",
            region_name,
            scope_label,
            scope_err,
        )
        return None


def _stored_region_key(storage, region_name: str) -> Optional[bytes]:
    """Provisioned transport key for a private ('$') region, or None."""
    get_keys = getattr(storage, "get_transport_keys", None)
    if not callable(get_keys):
        return None

    target = region_name.strip().lower()
    for record in get_keys() or []:
        if not isinstance(record, dict):
            continue
        if str(record.get("name", "")).strip().lower() != target:
            continue
        encoded = record.get("transport_key")
        if not encoded:
            return None
        try:
            import base64

            return base64.b64decode(encoded)
        except Exception as decode_err:
            logger.warning(
                "Invalid stored key for private region '%s': %s", region_name, decode_err
            )
            return None
    return None
