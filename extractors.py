from regex_helper import DATE_PATTERN, _TOTAL_PATTERNS, _JUNK_LINE, _JUNK_KEYWORD, _COMPANY_BOOST


def looks_like_company(line):
    if _COMPANY_BOOST.search(line):
        return True
    # company name usually have at least 2 words in receipts
    # and also all upper case
    # and prolly the name is more than 4 characters (just a arbitrary threshold maybe should study more)
    # just to prevent hand-written text at the top of receipt or bad ocr extraction from lib
    words = line.split()
    if len(words) >= 2 and line == line.upper() and len(line) >= 4 and line.replace(' ', '').isalpha():
        return True
    return False


def extract_vendor(lines):
    # this function helps us to prevent detecting hand-written text at the top of receipt or bad ocr extraction from lib
    # and considering them as vendor names
    candidates = []
    # 100% company name is in the first 10 lines
    # no space for them to write more than that at the top
    # and I don't think the ocr lib will include trash extraction more than more than 10 lines
    for line in lines[:10]:
        line = line.strip()
        if not line or _JUNK_LINE.match(line) or _JUNK_KEYWORD.search(line):
            continue
        if line.isdigit():
            continue
        if looks_like_company(line):
            return line
        candidates.append(line)
    return candidates[0] if candidates else None  # if all just fail get the first line anyway


def extract_date(ocr_text):
    ocr_text = ocr_text.replace('\r\n', '\n').replace('\r', '\n')
    match = DATE_PATTERN.search(ocr_text)
    return match.group(1).strip() if match else None


def extract_total(text):
    for pattern in _TOTAL_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            return matches[-1].group(matches[-1].lastindex)
    return None
