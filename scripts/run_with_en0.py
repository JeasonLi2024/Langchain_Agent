"""Run a script with gateway-bound TCP sockets routed via the physical interface.

Bypasses the Cisco AnyConnect tunnel (utun6) which blackholes TLS traffic to
10.3.19.2 (llm-gw.bupt.edu.cn). Covers BOTH sync (socket.create_connection)
and async (httpx/anyio/asyncio) paths by patching socket.socket.connect.

Only connections whose destination matches BIND_IF_HOSTS are bound; other
traffic (local milvus-lite loopback, remote Milvus via tunnel, DNS) untouched.

Usage:
    python scripts/run_with_en0.py <script.py> [args...]

Env:
    BIND_IF=en0              interface to use (default en0)
    NO_BIND_IF=1             disable the patch entirely
    BIND_IF_HOSTS=...        comma-separated targets (default llm-gw.bupt.edu.cn)
"""
import os
import runpy
import socket
import sys

IF = os.environ.get("BIND_IF", "en0")
HOSTS = [h.strip() for h in os.environ.get("BIND_IF_HOSTS", "llm-gw.bupt.edu.cn").split(",") if h.strip()]

if not os.environ.get("NO_BIND_IF"):
    try:
        IF_INDEX = socket.if_nametoindex(IF)
    except (OSError, ValueError):
        IF_INDEX = 0

    TARGET_IPS = set()
    for h in HOSTS:
        try:
            TARGET_IPS.add(socket.gethostbyname(h))
        except OSError:
            pass

    if IF_INDEX and TARGET_IPS:
        _orig_socket = socket.socket

        class BoundSocket(_orig_socket):
            def _maybe_bind_if(self, address):
                try:
                    ip = address[0] if isinstance(address, tuple) else address
                    if ip in TARGET_IPS:
                        v6 = self.family == socket.AF_INET6
                        level = socket.IPPROTO_IPV6 if v6 else socket.IPPROTO_IP
                        opt = 125 if v6 else 25
                        self.setsockopt(level, opt, IF_INDEX)
                except OSError:
                    pass

            def connect(self, address):
                self._maybe_bind_if(address)
                return super().connect(address)

            def connect_ex(self, address):
                self._maybe_bind_if(address)
                return super().connect_ex(address)

        socket.socket = BoundSocket
        print(f"[run_with_en0] sockets to {sorted(TARGET_IPS)} bound to {IF} (index {IF_INDEX})")
    else:
        print(f"[run_with_en0] interface {IF} or targets missing, normal routing")
else:
    print("[run_with_en0] binding disabled")

if len(sys.argv) < 2:
    sys.exit(__doc__)

sys.argv = sys.argv[1:]  # strip wrapper path: target script becomes argv[0], its args follow
runpy.run_path(sys.argv[0], run_name="__main__")
