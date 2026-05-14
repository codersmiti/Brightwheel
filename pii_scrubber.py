import re

def scrub_pii(text):
    # Mask email addresses
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', text)
    
    # Mask phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    
    # Mask common name patterns after keywords
    text = re.sub(r'(?i)(parent|teacher|director|staff|contact|name is|from)\s+[A-Z][a-z]+\s+[A-Z][a-z]+', 
                  r'\1 [NAME]', text)
    
    return text

if __name__ == "__main__":
    test = "Call me at 555-204-8811 or email srivera@sunshinelearning.org. Ask for Sandra Rivera."
    print(scrub_pii(test))