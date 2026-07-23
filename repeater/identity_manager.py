import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("IdentityManager")


class IdentityConfigurationError(RuntimeError):
    """A configured local identity cannot be represented safely."""


# The on-air destination hash is a single byte (the first byte of the public
# key), so every local identity collapses to a one-byte prefix.  Two identities
# that would occupy the *same* runtime map at that prefix cannot coexist; two in
# *different* maps can, because the receive path (see packet_router) offers a
# colliding packet to both candidates and MAC verification selects the true
# owner.  Companions register in RepeaterDaemon.companion_bridges (and a
# per-prefix SQLite namespace); repeater and room-server identities register in
# the helper handler maps (login/text/protocol_request), which are also keyed by
# the one-byte prefix.  So the namespaces are "companion" and "server".
_SERVER_TYPES = frozenset({"repeater", "room_server"})


def _routing_namespace(identity_type: str) -> str:
    """Return the one-byte-prefix routing namespace for an identity type.

    Two identities can share a prefix only when their namespaces differ.
    """
    return "companion" if identity_type == "companion" else "server"


@dataclass(frozen=True)
class IdentitySpec:
    """A parsed-but-unregistered local identity from configuration."""

    name: str
    identity: Any  # openhop_core LocalIdentity (or compatible)
    config: dict
    identity_type: str  # "repeater" | "room_server" | "companion"

    @property
    def label(self) -> str:
        return f"{self.identity_type}:{self.name}"

    @property
    def hash_byte(self) -> int:
        return self.identity.get_public_key()[0]

    @property
    def namespace(self) -> str:
        return _routing_namespace(self.identity_type)


class IdentityManager:
    def __init__(self, config: dict):
        self.config = config
        self.identities: Dict[int, Tuple[Any, dict, str]] = {}
        self.named_identities: Dict[str, Tuple[Any, dict, str]] = {}
        self.registered_hashes: Dict[int, str] = {}
        # Every identity registered at a one-byte prefix, so a same-namespace
        # collision can be detected even when several identities share the byte.
        self.hash_owners: Dict[int, List[Tuple[str, str]]] = {}

    def _same_namespace_owner(
        self, hash_byte: int, identity_type: str
    ) -> Optional[Tuple[str, str]]:
        """Return an already-registered ``(type, name)`` that shares both the
        one-byte prefix and the routing namespace of ``identity_type``, or
        ``None`` when the prefix is free within that namespace.
        """
        namespace = _routing_namespace(identity_type)
        for existing_type, existing_name in self.hash_owners.get(hash_byte, []):
            if _routing_namespace(existing_type) == namespace:
                return existing_type, existing_name
        return None

    def registration_error(self, name: str, identity, identity_type: str) -> Optional[str]:
        """Return a reason this identity cannot be registered, or ``None``.

        The on-air destination hash is only the first byte of the public key, so
        local routing and companion persistence are keyed by that byte.  A
        collision with an identity in the same routing namespace (companion vs
        companion, or server-side vs server-side) therefore cannot be
        represented safely and is rejected.  A collision across namespaces (a
        companion and a repeater/room server) is tolerated because the receive
        path offers the packet to both and MAC verification decides the owner.
        Names must also be unique because callers use them to locate the
        configured service.
        """
        hash_byte = identity.get_public_key()[0]

        clash = self._same_namespace_owner(hash_byte, identity_type)
        if clash is not None:
            existing_type, existing_name = clash
            namespace = _routing_namespace(identity_type)
            return (
                f"Identity '{name}' (hash=0x{hash_byte:02X}) conflicts with "
                f"existing {existing_type} '{existing_name}'; two {namespace} "
                "identities cannot share a one-byte public-key prefix"
            )

        if name in self.named_identities:
            existing_identity, _, existing_type = self.named_identities[name]
            existing_hash = existing_identity.get_public_key()[0]
            return (
                f"Identity name '{name}' is already registered for "
                f"{existing_type} (hash=0x{existing_hash:02X})"
            )

        return None

    def validate_specs(self, specs: Iterable[IdentitySpec]) -> None:
        """Raise ``IdentityConfigurationError`` on any name or unsafe hash
        collision, without mutating any state.

        Each spec is checked against the currently registered identities (via
        :meth:`registration_error`) and against the other specs in the batch.  A
        one-byte prefix collision is fatal only when both identities share a
        routing namespace (companion/companion or server/server); a companion
        colliding with a repeater or room server is tolerated and left to the
        MAC-verifying receive path.  Names must be unique because callers use
        them to locate the configured service.
        """
        batch_names: Dict[str, IdentitySpec] = {}
        batch_owners: Dict[int, List[IdentitySpec]] = {}

        for spec in specs:
            error = self.registration_error(spec.name, spec.identity, spec.identity_type)
            if error:
                raise IdentityConfigurationError(error)

            existing = batch_names.get(spec.name)
            if existing is not None:
                raise IdentityConfigurationError(
                    f"Local identity name '{spec.name}' conflicts with existing "
                    f"identity '{existing.label}'"
                )

            for other in batch_owners.get(spec.hash_byte, []):
                if other.namespace == spec.namespace:
                    raise IdentityConfigurationError(
                        f"Local identity '{spec.label}' (hash=0x{spec.hash_byte:02X}) "
                        f"conflicts with '{other.label}'; two {spec.namespace} "
                        "identities cannot share a one-byte public-key prefix"
                    )

            batch_names[spec.name] = spec
            batch_owners.setdefault(spec.hash_byte, []).append(spec)

    def validate_identity(self, name: str, identity, identity_type: str) -> bool:
        """Log and report whether an identity can be registered without mutation."""
        error = self.registration_error(name, identity, identity_type)
        if error:
            logger.error("Identity registration rejected: %s", error)
            return False
        return True

    def register_identity(self, name: str, identity, config: dict, identity_type: str):
        if not self.validate_identity(name, identity, identity_type):
            return False

        hash_byte = identity.get_public_key()[0]

        # A tolerated cross-namespace collision still deserves a warning: the
        # node functions (the receive path MAC-verifies the owner) but the
        # one-byte prefix is shared, which is worth surfacing in the log.
        for existing_type, existing_name in self.hash_owners.get(hash_byte, []):
            logger.warning(
                "Local identity '%s:%s' (hash=0x%02X) shares its one-byte "
                "public-key prefix with '%s:%s'; both remain usable because the "
                "receive path selects the owner by MAC verification",
                identity_type,
                name,
                hash_byte,
                existing_type,
                existing_name,
            )

        # Keep the first identity registered at a prefix as the primary for the
        # byte-keyed lookups; named_identities holds every identity.
        self.identities.setdefault(hash_byte, (identity, config, identity_type))
        self.registered_hashes.setdefault(hash_byte, f"{identity_type}:{name}")
        self.named_identities[name] = (identity, config, identity_type)
        self.hash_owners.setdefault(hash_byte, []).append((identity_type, name))

        logger.info(
            f"Identity registered: name={name}, hash=0x{hash_byte:02X}, type={identity_type}"
        )
        return True

    def get_identity_by_hash(self, hash_byte: int) -> Optional[Tuple[Any, dict, str]]:
        return self.identities.get(hash_byte)

    def get_identity_by_name(self, name: str) -> Optional[Tuple[Any, dict, str]]:
        return self.named_identities.get(name)

    def has_identity(self, hash_byte: int) -> bool:
        return hash_byte in self.identities

    def list_identities(self) -> list:
        # Iterate named_identities so every registered identity is listed, even
        # when two share a one-byte prefix (only one occupies self.identities).
        identities = []
        for name, (identity, config, id_type) in self.named_identities.items():
            hash_byte = identity.get_public_key()[0] if identity else None
            identities.append(
                {
                    "hash": f"0x{hash_byte:02X}" if hash_byte is not None else "N/A",
                    "name": f"{id_type}:{name}",
                    "type": id_type,
                    "address": identity.get_address_bytes().hex() if identity else "N/A",
                    "public_key": identity.get_public_key().hex() if identity else None,
                }
            )
        return identities

    def has_identity_type(self, identity_type: str) -> bool:
        return any(id_type == identity_type for _, _, id_type in self.named_identities.values())

    def get_identities_by_type(self, identity_type: str) -> list:
        results = []
        for name, (identity, config, id_type) in self.named_identities.items():
            if id_type == identity_type:
                results.append((name, identity, config))
        return results
