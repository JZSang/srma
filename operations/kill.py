from modal_references import status_tracker_global_dictionary


class KillException(Exception):
    pass


# TODO: use function object_id to kill instead without losing the progress made so far
def kill_impl(mode):
    if mode != "test" and mode != "gptcot":
        raise ValueError(f"Invalid mode {mode}")
    status_tracker_global_dictionary[mode + "kill"] = True
    return True
