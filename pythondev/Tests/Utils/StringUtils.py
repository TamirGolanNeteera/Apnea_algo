def join(lis: list, sep=' ') -> str:
    return sep.join([str(x) for x in lis])


def string_match_out_of_list(string, list_strings):
    for x in list_strings:
        if x in string:
            return x
    return False


def subtract_strings(minuend, subtrahend):
    if subtrahend in minuend:
        index = minuend.find(subtrahend)
        return minuend[:index]
    else:
        return minuend
