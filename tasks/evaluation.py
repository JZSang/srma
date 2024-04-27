def is_excluded(text, test_abstract):
    if text is None:
        raise Exception(f"Invalid case: LLM did not print an answer: {test_abstract}")
    # search area to the very end, maybe about 500 characters
    text = text[-500:]
    if "XXX" in text and "YYY" in text:
        if "YYY" in text.split("XXX")[-1]:
            # if YYY is after XXX, then it is included
            return "included"
        else:
            return "excluded"
    elif "XXX" in text:
        return "excluded"
    elif "YYY" in text:
        return "included"
    else:
        raise Exception(
            f"Invalid case: LLM did not print an answer: {text} {test_abstract}"
        )