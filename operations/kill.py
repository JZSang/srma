from modal_references import stub


class KillException(Exception):
    pass


# TODO: use function object_id to kill instead without losing the progress made so far
def kill_impl(mode):
    if mode != "test" and mode != "gptcot":
        raise ValueError(f"Invalid mode {mode}")
    stub.status_tracker[mode + "kill"] = True
    return True
