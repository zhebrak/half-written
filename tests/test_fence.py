"""The exact fence (halfwritten/fence.py): the engine's own count decides when a pause's old tokens are all in, and
one cut sequence serves the controller and the session gate."""
import asyncio


def test_the_exact_fence_reports_a_count_that_never_settles():
    """wait_delivered: a count that keeps rising (the engine still generating after its pause) is not a cut; one that
    settles is, however long the controller itself stalled first."""
    from halfwritten.fence import wait_delivered

    async def main(rising: bool):
        loop = asyncio.get_running_loop()
        state = {"gen": 100, "got": 0}

        async def feed():
            while True:
                await asyncio.sleep(0.005)
                state["got"] = min(state["got"] + 10, state["gen"])
                if rising:
                    state["gen"] += 10

        async def read():
            return state["gen"]

        task = asyncio.create_task(feed())
        out = await wait_delivered(state["gen"], read, lambda: state["got"], loop.time, timeout_s=0.3, max_s=0.6)
        task.cancel()
        return out, state

    (count, settled), state = asyncio.run(main(False))
    assert settled and count == 100 == state["got"]
    (count, settled), _ = asyncio.run(main(True))
    assert not settled


def test_the_exact_fence_waits_longer_once_a_server_is_behind():
    """One saturated API server (D on one server, two H100s): the count rose after its read at the ack at 45 of 55
    syncs. A server behind delivers and counts a step late, so once the count has moved, a 20 ms agreement is not
    enough: here the server's last step lands 50-60 ms after the counts first agree."""
    from halfwritten.fence import wait_delivered

    async def main():
        loop = asyncio.get_running_loop()
        t0, state = loop.time(), {"gen": 100, "got": 90}

        async def server():
            await asyncio.sleep(0.01)
            state["got"] = 100
            await asyncio.sleep(0.04)
            state["got"] = 110  # its last step's outputs, then their count
            await asyncio.sleep(0.01)
            state["gen"] = 110

        async def read():
            return state["gen"]

        task = asyncio.create_task(server())
        out = await wait_delivered(90, read, lambda: state["got"], loop.time)  # read at the ack: 90
        await task
        return out, loop.time() - t0

    (count, settled), took = asyncio.run(main())
    assert settled and count == 110 and took < 0.5


def test_the_cut_reads_the_count_alongside_the_push_and_verifies_before_it_settles():
    """exact_cut, the sequence the controller and the canary share: the count is read while the push runs (the
    engine is paused, so it is final), `between` sees the push's result before any waiting, and the wait ends only
    once delivery matches the count and holds."""
    from halfwritten.fence import exact_cut

    async def main():
        loop = asyncio.get_running_loop()
        order, state = [], {"gen": 50, "got": 40}

        async def push():
            order.append("push starts")
            await asyncio.sleep(0.03)
            order.append("push ends")
            return "pushed-v7"

        async def read():
            order.append(f"read {state['gen']}")
            return state["gen"]

        async def between(pushed):
            order.append(f"between {pushed}")

        async def deliver():
            await asyncio.sleep(0.05)
            state["got"] = 50
            order.append("delivered")

        task = asyncio.create_task(deliver())
        out = await exact_cut(push(), read, lambda: state["got"], loop.time, between=between)
        await task
        return out, order

    (pushed, first, final, settled, waited), order = asyncio.run(main())
    assert (pushed, first, final, settled) == ("pushed-v7", 50, 50, True) and waited > 0.02
    assert order[:2] == ["push starts", "read 50"]  # the count read alongside the push, not after it
    assert order.index("between pushed-v7") == order.index("push ends") + 1  # verified before the wait
    assert order.index("delivered") < len(order) - 1 and order[-1] == "read 50"  # the second read, once delivered
