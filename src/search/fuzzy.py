
from thefuzz import fuzz
import json
import time # Import the time module

def find_best_menu_item_from_file(menu_file_path, search_name, threshold=70):
    """
    Finds the best matching menu item from a JSON menu file.

    Args:
        menu_file_path (str): The path to the JSON file containing the menu data.
        search_name (str): The name of the item to search for.
        threshold (int): The minimum similarity score (0-100) to consider a match.

    Returns:
        tuple: (best_match_item_dict, highest_score) or (None, 0) if no suitable match.
    """
    try:
        with open(menu_file_path, 'r', encoding='utf-8') as f:
            menu = json.load(f)
    except FileNotFoundError:
        print(f"Error: Menu file not found at '{menu_file_path}'")
        return None, 0
    except json.JSONDecodeError as e:
        # This error message is crucial. It will tell you where the JSON is malformed.
        print(f"Error: Invalid JSON in menu file '{menu_file_path}'.")
        print(f"Details: {e}")
        print(f"Please validate your '{menu_file_path}' file. You can use an online JSON validator.")
        return None, 0
    except Exception as e:
        print(f"An unexpected error occurred while reading the menu file: {e}")
        return None, 0

    best_match_item = None
    highest_score = 0

    if "results" not in menu or not isinstance(menu["results"], list):
        print("Warning: 'results' key not found or not a list in menu data.")
        return None, 0

    for category in menu["results"]:
        if "items" in category and isinstance(category["items"], list):
            for item in category["items"]:
                item_title = item.get("title")
                if item_title:  # Ensure title exists
                    score = fuzz.ratio(search_name.lower(), item_title.lower())
                    
                    if score > highest_score:
                        highest_score = score
                        best_match_item = item
    
    if highest_score >= threshold:
        return best_match_item, highest_score
    else:
        return None, highest_score

# --- Example Usage ---
if __name__ == "__main__":
    menu_file = "menu.json" 

    queries = [
        "Mint Oreo Chiller",
        "clasic nova lox",      # Misspelled
        "oreginal clasic",      # Misspelled and part of "Original Classic"
        "hash brown",
        "fruity rice",
        "Cheesy Hashbrown Roll", # Specific item, should get high score
        "Breakfast Sandwich"     # More general, might match a category or multiple items less perfectly
    ]

    print("--- Finding Menu Items (fuzz.ratio from file) ---")
    for search_query in queries:
        start_time = time.perf_counter() # Start timer
        matched_item, score = find_best_menu_item_from_file(menu_file, search_query, threshold=60) # Lowered threshold for demo
        end_time = time.perf_counter()   # End timer
        duration_ms = (end_time - start_time) * 1000 # Calculate duration in milliseconds

        print(f"\nSearching for: '{search_query}'")
        if matched_item:
            print(f"  Best match: '{matched_item['title']}'")
            print(f"  Price: {matched_item.get('price', 'N/A')}")
            print(f"  Score: {score}%")
            
            modifier_groups = matched_item.get("modifier_groups")
            if modifier_groups and isinstance(modifier_groups, list) and len(modifier_groups) > 0:
                print("  Modifier Groups:")
                for group_idx, group in enumerate(modifier_groups):
                    group_title = group.get('title', f'Unnamed Group {group_idx+1}')
                    print(f"    - {group_title}:")
                    if "modifiers" in group and isinstance(group["modifiers"], list):
                        for mod_idx, modifier in enumerate(group["modifiers"]):
                            mod_title = modifier.get('title', f'Unnamed Modifier {mod_idx+1}')
                            mod_price = modifier.get('price', 0) 
                            print(f"      - {mod_title} (Price: {mod_price})")
                    else:
                        print("      (No specific modifiers listed in this group)")
            else:
                print("  No modifier groups for this item.")
        else:
            print(f"  No good match found. Highest score was {score}%.")
        
        print(f"  Time taken: {duration_ms:.2f} ms")