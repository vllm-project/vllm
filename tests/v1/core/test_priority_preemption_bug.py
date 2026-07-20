from vllm.v1.core.sched.scheduler import RequestStatus


def test_priority_scheduler_preempt_skipped_request():
    # Importing helper functions from the reproduction script
    from repro import build_scheduler, make_request, mock_output

    scheduler = build_scheduler()

    # R1: Add request A (Worst priority)
    A = make_request("A", num_tokens=200, priority=9, arrival_time=1.0)
    scheduler.add_request(A)
    out = scheduler.schedule()
    scheduler.update_from_output(out, mock_output(out))

    # R2: Add request B (Best priority)
    B = make_request("B", num_tokens=15, priority=0, arrival_time=2.0)
    scheduler.add_request(B)
    out = scheduler.schedule()
    scheduler.update_from_output(out, mock_output(out))

    # R3: Add request C (Middle priority)
    C = make_request("C", num_tokens=1, priority=1, arrival_time=3.0)
    scheduler.add_request(C)
    out = scheduler.schedule()
    scheduler.update_from_output(out, mock_output(out))

    # R4: Trigger preemption
    out = scheduler.schedule(throttle_prefills=True)

    is_c_scheduled = "C" in out.num_scheduled_tokens

    assert is_c_scheduled, (
        "Bug present: Request C was silently skipped during scheduling "
        "because the req_index was not decremented after preempting A."
    )
    assert C.status == RequestStatus.RUNNING
