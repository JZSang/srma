
from modal_references import stub

def get_status_impl(mode):
    return stub.status_tracker[mode].__dict__