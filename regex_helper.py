# same regex from level 2
# probably the only thing new is
# the vendor regex because I found
# bad extractions during testing
import re

DATE_PATTERN = re.compile(
    r'\b('
    r'\d{4}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])'
    r'|(?:0[1-9]|[12]\d|3[01])(?:0[1-9]|1[0-2])\d{4}'
    r'|\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}(?=[^\d]|$)'
    r'|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}'
    r'|\d{1,2}[\/\-\.\s][A-Za-z]{3}[\/\-\.\s]\d{2,4}'
    r'|[A-Za-z]{3}\.?\s+\d{1,2},?\s+\d{4}'
    r')',
    re.IGNORECASE
)


_NUM = r'[\d]{1,3}(?:,\d{3})*(?:\.\d{2})|\d+\.\d{2}'

_TOTAL_PATTERNS = [
    # re.compile(rf'(?i)(?:rounded|inclus[a-z]*)\b[^\d\n%]*[\r\n]*\s*(?:RM\s*)?({_NUM})(?!\s*%)'),
    re.compile(rf'(?i)(?:rounded|incl[a-z]*)\b[^\d\n%]*[\r\n]*\s*(?:RM\s*)?({_NUM})(?!\s*%)'),

    # re.compile(rf'(?im)^(?!.*(?:%|GST|TAX))(?<!\w)total\b[^\n\d]*?(?:RM\s*)?({_NUM})'),
    re.compile(rf'(?im)^[ \t]*(?!.*(?:%|GST|TAX))total\b[^\n\d]*?(?:RM\s*)?({_NUM})'),

    # re.compile(rf'(?im)^(?!.*(?:%|GST|TAX))(?<!\w)total\b[^\n]*\n\s*(?:RM\s*)?({_NUM})'),
    re.compile(rf'(?im)^[ \t]*(?!.*(?:%|GST|TAX))total\b[^\n]*\n\s*(?:RM\s*)?({_NUM})'),

    # re.compile(rf'(?is)total\s+incl.*?(?:RM\s*)?({_NUM})'),
    re.compile(rf'(?is)total\s*\(?incl.*?(?:RM\s*)?({_NUM})'),

    re.compile(rf'(?i)net\s+amt\b.*?({_NUM})'),
]

# vendor names are nothing like the following
# so we can easily get these out if found
_JUNK_LINE = re.compile(
    r'(?i)^('
    r'\s*'
    r'|[a-z0-9]{1,2}'
    r'|tax\s*invoice'
    r'|cash\s*receipt'
    r'|receipt'
    r'|invoice'
    r'|official\s*receipt'
    r'|retail\s*invoice'
    r')$'
)

_JUNK_KEYWORD = re.compile(
    r'(?i)\b(date|time|tel|gst|vat|no\.?)\b'
)

# _COMPANY_BOOST = re.compile(
#     r'(?i)\b(sdn|bhd|plt|sdn\.?\s*bhd|enterprise|trading|holdings?|'
#     r'industries|services|mart|shop|store|racking|pharmacy|clinic|'
#     r'supermarket|hypermarket|restaurant|hardware|berhad)\b'
# )
# after studying the receipts and finding the patterns that company names might have
# this should work
_COMPANY_BOOST = re.compile(
    r'(?i)\b('
    r'sdn\.?\s*bhd|berhad|bhd|plt|'
    r'inc|llc|ltd|corp|co|company|'
    r'group|holdings?|international|'
    r'enterprise|trading|industries|services|'
    r'mart|shop|store|pharmacy|clinic|restaurant'
    r')\b'
)