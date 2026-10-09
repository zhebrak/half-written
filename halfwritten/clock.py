"""Virtual time for asyncio, so the real controller runs against FakeEngine without waiting.

VirtualTimeLoop is a stock SelectorEventLoop whose selector, instead of blocking until the next timer, moves the
loop's clock forward to it. Code reads time only through loop.time() and waits only through the loop (sleep,
wait_for, futures), so a 45-minute run takes as long as its callbacks do, and two runs with the same inputs produce
the same trace. When nothing is ready and no timer is pending the real loop would block forever; here that raises
SimDeadlock, so a controller hang fails a test instead of hanging it.
"""
import asyncio
import selectors


class SimDeadlock(RuntimeError):
    pass


class _VirtualSelector(selectors.BaseSelector):
    """Delegates registration (the loop's self-pipe) to a real selector; select() polls it without blocking and
    advances the loop's clock by the timeout the loop asked for."""

    def __init__(self, loop: "VirtualTimeLoop"):
        self._loop = loop
        self._real = selectors.DefaultSelector()

    def register(self, fileobj, events, data=None):
        return self._real.register(fileobj, events, data)

    def unregister(self, fileobj):
        return self._real.unregister(fileobj)

    def modify(self, fileobj, events, data=None):
        return self._real.modify(fileobj, events, data)

    def select(self, timeout=None):
        events = self._real.select(0)
        if events:
            return events
        if timeout is None:
            raise SimDeadlock(f"no ready callback and no timer at t={self._loop.time():.6f}")
        if timeout > 0:
            self._loop._now += timeout
        return []

    def close(self):
        self._real.close()

    def get_map(self):
        return self._real.get_map()


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    def __init__(self, start: float = 0.0):
        self._now = float(start)
        super().__init__(selector=_VirtualSelector(self))
        self._clock_resolution = 1e-9

    def time(self) -> float:
        return self._now


def run_sim(main, start: float = 0.0):
    """Runs coroutine `main` to completion in virtual time and returns its result."""
    with asyncio.Runner(loop_factory=lambda: VirtualTimeLoop(start)) as runner:
        return runner.run(main)
