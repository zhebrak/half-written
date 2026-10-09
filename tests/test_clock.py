import asyncio
import time

import pytest

from halfwritten.clock import SimDeadlock, run_sim


def test_an_hour_of_sleep_takes_no_real_time():
    async def main():
        loop = asyncio.get_running_loop()
        await asyncio.sleep(3600)
        await asyncio.gather(asyncio.sleep(1), asyncio.sleep(2))
        return loop.time()

    t = time.monotonic()
    assert run_sim(main()) == pytest.approx(3602)
    assert time.monotonic() - t < 1


def test_timers_fire_in_order_at_their_virtual_times():
    async def main():
        loop, seen = asyncio.get_running_loop(), []
        for d in (0.3, 0.1, 0.2):
            loop.call_later(d, lambda d=d: seen.append((d, round(loop.time(), 9))))
        await asyncio.sleep(1)
        return seen

    assert run_sim(main()) == [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)]


def test_wait_for_times_out_in_virtual_time():
    async def main():
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.Event().wait(), 900)
        return asyncio.get_running_loop().time()

    assert run_sim(main()) == pytest.approx(900)


def test_a_hang_is_a_deadlock_not_a_hung_test():
    async def main():
        await asyncio.get_running_loop().create_future()

    with pytest.raises(SimDeadlock):
        run_sim(main())
