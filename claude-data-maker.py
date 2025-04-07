import anthropic
import csv
import json  # Required for parsing JSON responses
import os
import time
import datetime
import shutil


client = anthropic.Anthropic(
    # defaults to 
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

out_file = "training_data_new.csv"


def log_error(e,filename, log_file,func_name):
    """Logs errors with timestamp and traceback to a .txt file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, "a") as file:
        file.write(f"\n[{timestamp}] ERROR:\n")
        file.write(filename + "      " +str(e) + "          " + func_name + "\n \n" )
        file.write("-" * 50 + "\n")  # Separator for readability


def move_file(src_path, dest_folder):
    """
    Moves a file from src_path to dest_folder.
    
    Parameters:
        src_path (str): The full path of the file to move.
        dest_folder (str): The destination folder.
    
    Returns:
        str: The new file path if successful, else an error message.
    """
    try:
        # Ensure destination folder exists
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        # Move the file
        new_path = shutil.move(src_path, dest_folder)
        return f"File moved to: {new_path}"
    
    except Exception as e:
        return f"Error moving file: {e}"



def get_queries(product,filename, retry_count=0, max_retries=5):
    log_error("File accessed",filename,"looked_at.txt","get_queries")
    text = f"Generate a list of 20 search terms and their category but loosely based on the following product or a category that this product belongs to:\n {product}\n. Only provide the list, DO NOT GIVE ANY INTRODUCTORY TEXT. The only content in the reply should be the list of (search terms, category)."
    try:
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=8192,
            temperature=1,
            system=" \n\n---\n\nYou are a data analyst with 10 years of experience working on the following: **Search Term Classification System**  \n\n### **Task:**  \nClassify search terms as **\"Direct Embedding Search\"** or **\"Keyword Extraction Needed\"** based on specific criteria.  \n\n### **Classification Criteria**  \n\n#### **Direct Embedding Search**  \nApplies to search terms that:  \n- Contain **specific, concrete descriptors** (color, material, style, form)  \n- Are **self-contained** and don’t require external context  \n- Have direct **visual or textual embedding potential**  \n\n✅ **Examples:** \"navy blue suede sneakers\", \"rustic wooden dining table\", \"sleek black gaming laptop\"  \n\n#### **Keyword Extraction Needed**  \nApplies to search terms that:  \n- Contain **abstract qualities, contexts, or usage scenarios**  \n- Have **relative attributes** (price, trends, popularity)  \n- Include **questions, comparisons, named entities, or intent-driven queries**  \n\n❌ **Examples:** \"top-rated hiking boots\", \"how to choose a laptop\", \"best gifts for new parents\"  \n\n### **Classification Rules**  \n\n1. **Mixed queries:** Classify by dominant intent.  \n   - \"cheap stainless steel watch\" → **Direct Embedding**  \n   - \"best watches for business professionals\" → **Keyword Extraction**  \n\n2. **Automatic Keyword Extraction triggers:**  \n   - Time-based: **\"latest fashion trends\"**  \n   - Usage-based: **\"best shoes for travel\"**  \n   - Comparisons: **\"most durable winter coat\"**  \n   - Intent-driven: **\"How to...\", \"What is...\", \"Best for...\"**  \n   - DIY/process: **\"materials for candle making\"**  \n\n3. **Ambiguous cases:** Mark as **\"For Review\"**  \n\n### **Output Format**  \n- Return results as an **array**: `[[search term, classification]]`  \n- Do **not** use the given examples directly—generate similar but unique queries  \n- **ONLY** output the array, without extra text  \n \nExample Output Format:\n[\n  [\"colorful LED desk lamp\", \"Direct Embedding Search\"],\n  [\"how to organize a home office\", \"Keyword Extraction Needed\"],\n  [\"lightweight travel backpack\", \"Direct Embedding Search\"],\n  [\"best budget-friendly laptops\", \"Keyword Extraction Needed\"]\n]\n\n\n---\n\n. You reply to any prompts you receive only with a python list",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]
        )
    except Exception as e:
        if(retry_count<max_retries):
            print("There's been an issue boss.\n")
            print(f"The server said {e}\n")
            print("I will try again in 10 seconds\n")
            time.sleep(10)
            return get_queries(product,filename,retry_count+1)
        else:
            print("Sorry boss, I give up\n")
            log_error("Gave up trying the server",filename,"error_log.txt","get_queries")
            return [] 
    # Extract response text
    else:
        response_text = message.content[0].text

    try:
        # Convert JSON string into a Python list
        parsed_data = json.loads(response_text)
        if isinstance(parsed_data, list):
            print("\n That worked. I've parsed the data\n")
            return parsed_data
        else:
            raise ValueError("API response is not a list")
    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        print(message.content)
        log_error(e,filename,"error_log.txt","get_queries")
        get_queries(product,filename)
        return []





def write_to_csv(data, csv_file,filename):
    # Check if file exists to avoid writing headers multiple times
    if(len(data)>10):
        try:
            with open(csv_file, "r", encoding="utf-8") as file:
                file_exists = True
        except FileNotFoundError:
            file_exists = False

    # Writing to CSV in append mode
        with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

        # Write headers only if file doesn't exist
            if not file_exists:
                writer.writerow(["Search Query", "Classification"])

        
            for row in data:
                writer.writerow(row)
        print(f"CSV file '{csv_file}' updated successfully! ✅")
        return True
    else:
        print("There was no data to write")
        log_error("No data to write",filename,"error_log.txt","write_to_csv")
        return False   
    
    



def get_json_titles(folder_path):
    """Iterates through all JSON files in a folder, extracts and prints the 'title' property."""
    files = os.listdir(folder_path)
    # Iterate through all files in the folder
    for filename in files:
        if filename.endswith(".json"):  # Ensure it's a JSON file
            file_path = os.path.join(folder_path, filename)
            try:
                flag = False
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)  # Load JSON content
                    title = data.get("title", "No title found")  # Get 'title' property
                    flag= write_to_csv(get_queries(title,filename),out_file,filename)                      
                if flag:
                    print(move_file(file_path, "sample_processed"))
                    flag = False
                else:
                    print(f"Well that was a bummer for {filename}")
                    log_error("Couldn't do anything with",filename,"error_log.txt","get_json_titles")    
                time.sleep(3)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading {filename}: {e}")
                log_error(e,filename,"error_log.txt","print_json_titles")
        files = os.listdir(folder_path)    





# # Get queries and ensure correct parsing
# listoflists = get_queries("Gelatin Sticks")

# # Ensure valid data before writing
# if listoflists:
#     write_to_csv(listoflists, out_file)
# else:
#     print("No valid data to write.")

get_json_titles("sample_set")


# get_queries("coton candy", "output.txt")
