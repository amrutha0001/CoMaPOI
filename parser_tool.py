import re
import json
import logging

def extract_predicted_pois(content, top_k, key_name = 'next_poi_id'):
    """
    Extract 'next_poi_id' list from response content.
    Returns at most top_k POI IDs, filtering out non-numeric data.

    Args:
        content: Response content to parse
        top_k: Maximum number of POIs to return
        key_name: Key name to extract (default: 'next_poi_id')

    Returns:
        list: List of POI IDs
    """
    # Only print the first 100 characters of content to avoid cluttering the console
    if isinstance(content, str) and len(content) > 100:
        print(f"content (truncated): {content[:100]}...")
    else:
        print(f"content: {content}")

    poi_ids = []

    # Try using v2 parsing method
    try:
        if isinstance(content, dict):
            content_json = content
        else:
            content_json = json.loads(content)

        if 'next_poi_id' in content_json:
            next_poi_ids = content_json[key_name]
            for poi_id in next_poi_ids:
                if isinstance(poi_id, str):
                    match = re.search(r'\b(\d+)\b', poi_id)
                    if match:
                        poi_ids.append(match.group(1))
                elif isinstance(poi_id, (int, float)):
                    poi_ids.append(str(poi_id))
            return poi_ids[:top_k]  # Return the first top_k valid POI IDs

        else:
            pass
            #logging.error(f"Key {key_name} not found.")

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        #logging.error(f"v2 parsing error: {e}, trying v3 parsing method.")

        # Try using v3 parsing method
        try:
            if isinstance(content, dict):
                content_str = json.dumps(content)
            else:
                content_str = content

            pattern = r'"next_poi_id"\s*:\s*\[([^\]]+)\]'
            match = re.search(pattern, content_str)

            if match:
                ids_str = match.group(1)
                ids_list = re.split(r',\s*', ids_str)

                for id_str in ids_list:
                    # Extract numeric part
                    num_match = re.search(r'\b(\d+)\b', id_str)
                    if num_match:
                        poi_ids.append(num_match.group(1))

                return poi_ids[:top_k]  # Return the first top_k valid POI IDs

            else:
                # Try using v4 parsing method (markdown code block)
                pattern = r'```(?:json)?\s*{[^}]*"next_poi_id"\s*:\s*\[([^\]]+)\][^}]*}\s*```'
                match = re.search(pattern, content_str)

                if match:
                    ids_str = match.group(1)
                    ids_list = re.split(r',\s*', ids_str)

                    for id_str in ids_list:
                        # Extract numeric part
                        num_match = re.search(r'\b(\d+)\b', id_str)
                        if num_match:
                            poi_ids.append(num_match.group(1))

                    return poi_ids[:top_k]  # Return the first top_k valid POI IDs

                else:
                    # Try using v5 parsing method (direct number extraction)
                    numbers = re.findall(r'\b\d+\b', content_str)
                    return numbers[:top_k]  # Return the first top_k numbers found

        except Exception as e2:
            #logging.error(f"v3 parsing error: {e2}, using fallback method.")

            # Fallback method: extract all numbers
            if isinstance(content, str):
                numbers = re.findall(r'\b\d+\b', content)
                return numbers[:top_k]  # Return the first top_k numbers found

    # If all methods fail, return empty list
    return poi_ids[:top_k]


def extract_json_from_markdown(text):
    """
    Extract JSON content from markdown code blocks.

    Args:
        text: Markdown text containing code blocks

    Returns:
        str: Extracted JSON content or original text if no code blocks found
    """
    # Look for markdown code blocks (```json ... ```)
    pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(pattern, text)

    if matches:
        # Return the content of the first code block
        return matches[0].strip()

    # If no code blocks found, return the original text
    return text


def extract_poi_ids_from_text(text, top_k=10):
    """
    Extract POI IDs from text using various methods.

    Args:
        text: Text to extract POI IDs from
        top_k: Maximum number of POIs to return

    Returns:
        list: List of POI IDs
    """
    poi_ids = []

    # Try to extract JSON from markdown
    json_text = extract_json_from_markdown(text)

    # Try to parse as JSON
    try:
        data = json.loads(json_text)
        if isinstance(data, dict) and 'next_poi_id' in data:
            # Extract POI IDs from next_poi_id field
            for poi in data['next_poi_id']:
                if isinstance(poi, (int, float)):
                    poi_ids.append(str(int(poi)))
                elif isinstance(poi, str):
                    # Extract numeric part
                    match = re.search(r'\b(\d+)\b', poi)
                    if match:
                        poi_ids.append(match.group(1))
    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract numbers directly
        numbers = re.findall(r'\b\d+\b', text)
        poi_ids = numbers

    return poi_ids[:top_k]
