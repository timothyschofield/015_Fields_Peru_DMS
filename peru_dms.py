"""
File : peru_dms.py

Author: Tim Schofield
Date: 03 August 2024

ocr_column_names = [ 
        ("DarCollector","Collector Name"), 
        ("CollectionTeam","Collection Team"), 
        ("CollectionNumberPrefix","Collection Number Prefix"), 
        ("CollectionNumber","Collection Number"), 
        ("CollectionNumberSuffix","Collection Number Suffix"), 
        ("CollectionNumberText","Collection Number Verbatim"), 
        ("ColDateVisitedFrom","Collection Date From"), 
        ("ColDateVisitedTo","Collection Date To"), 
        ("ColVerbatimDate","Collection Date Verbatim"), 
        ("DarCollectionNotes","Collection Notes"), 
        ("DarContinent","Continent"), 
        ("DarCountry","Country"), 
        ("DarStateProvince","Province"), 
        ("DarCounty","County"),
        ("DarLocality","Locality Description"), 
        ("Township_tab","Township_tab"), 
        ("Range_tab","Range_tab"), 
        ("Section_tab", "Section_tab"),
        ("DarMinimumElevationMeters","Minimum Elevation in Meters"), 
        ("DarMaximumElevationMeters","Maximum Elevation in Meters"), 
        ("MinimumElevationFeet","Minimum Elevation in Feet"), 
        ("MaximumElevationFeet","Maximum Elevation in Feet"),
        ("DarLatitudeDecimal","Latitude Decimal"), 
        ("DarLongitudeDecimal","Longitude Decimal"), 
        ("LatitudeDMS","Latitude (Degrees Minutes Seconds)"), 
        ("LongitudeDMS","Longitude (Degrees Minutes Seconds)"), 
        ("DarGeodeticDatum","Geodetic Datum"), 
        ("DarGeorefMethod","Geo Reference Method"), 
        ("DarCoordinateUncertaintyInMeter","Coordinate Uncertainty in Meters"), 
        ("ColLocationNotes","Collection Location Notes"),
        ("FeaCultivated? (Y/N)","Cultivated"), 
        ("FeaPlantFungDescription","Plant Description"), 
        ("FeaFrequency","Plant Frequency"), 
        ("HabHabitat","Plant Habitat"), 
        ("HabVegetation","Plant Forest Type"), 
        ("HabSubstrate","Plant Substrate"), 
        ("SpeOtherSpecimenNumbers_tab","SpeOtherSpecimenNumbers_tab"), 
        ("MyOcrText", "OCR Text")]



prompt = (
    f"Read this herbarium sheet and extract all the text you can"
    f"The herbarium sheet may sometimes use Spanish, French or German"
    f"Go through the text you have extracted and return data in JSON format with {keys_concatenated} as keys"
    f"Use exactly {keys_concatenated} as keys"
    
    f"Use the English spelling for Country e.g. 'Brazil' not 'Brasil'"
    
    f"Return the text you have extracted in the field 'OCR Text'"
    
    f"'Collection Team' should contain other people involved in collecting the specimen"
    
    f"The 'Collection Date To' and 'Collection Date From' should have the format YYYY-MM-DD"
    f"If there is only one date then fill in 'Collection Date To' and 'Collection Date From' with the same value"
    
    f"Infer the Continent field from the Country e.g. If the Country is 'Belize' then the Continent field is 'Central America', if the Country is 'Costa Rica' the the Continent field is 'South America'"
    f"If no Country is mentioned then infer it from the Continent, Province, County and Locality Description"
    
    f"If Latitude and Longitude are not mentioned in the text then infer them from the Country, Province, County and Locality Description"
    f"Put the infered Latitude and Longitude in the 'Latitude (Degrees Minutes Seconds)' 'Longitude (Degrees Minutes Seconds)' 'Latitude Decimal' and 'Longitude Decimal' fields"
    f"If Latitude and Longitude have been inferred fill in the 'Coordinate Uncertainty In Meters' with an estimate of the accuracy and 'Geo Reference Method' with 'Estimated from locality description'"
    
    f"If a single elevation or altitude is mentioned fill in both the 'Minimum Elevation (Meters)' and 'Maximum Elevation (Meters)' with the same value"
    f"If there is elevation information in Meters then do a conversion to feet and store the conversion in 'Minimum Elevation (Feet)' and 'Maximum Elevation (Feet)'"
    
    f"For 'Plant Frequency' field look for words like abundant, cccasional, common, frequent or rare"
    f"For 'Plant Habitat' field put what type of environment the plant grows in e.g. forest, scrub, rocky hillside"
    f"For 'Plant Substrate' field look for what the plant grows on e.g. on rotting log, on damp rock, on bark"
    f"For 'Plant Description' field put a description of the plant e.g. Shrub 4m high, flowers white, fruit orange"
    
    f"If a plant is cultivated put 'Yes' in the 'Cultivated' field, otherwise put 'No'"
    
    f"If you find 'INB' followed by a number add INB and the number to SpeOtherSpecimenNumbers_tab"
    
    f"If you can not find a value for a key return value 'none'"
)


from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "system",
      "content": "You will be provided with a tweet, and your task is to classify its sentiment as positive, neutral, or negative."
    },
    {
      "role": "user",
      "content": "I loved the new Batman movie!"
    }
  ],
  temperature=0.7,
  max_tokens=64,
  top_p=1
)


"""
import openai
from openai import OpenAI
from dotenv import load_dotenv
from helper_functions_ny_herbarium import get_file_timestamp, is_json, make_text_payload, clean_up_ocr_output_json_content, are_keys_valid, get_headers, save_dataframe_to_csv

import requests
import os
from pathlib import Path 
import numpy as np
import pandas as pd
import time
from datetime import datetime
import json
import sys
print(f"Python version {sys.version}")

MODEL = "gpt-4o" # Context window of 128k max_tokens 4096

load_dotenv()
try:
    my_api_key = os.environ['OPENAI_API_KEY']          
    client = OpenAI(api_key=my_api_key)
except Exception as ex:
    print("Exception:", ex)
    exit()


input_folder = "peru_dms_input"
input_file = "Peru_field_from_Frank.csv"
input_path = Path(f"{input_folder}/{input_file}")

output_folder = "peru_dms_output"

project_name = "peru_dms" 

batch_size = 3 # saves every
time_stamp = get_file_timestamp()

df_to_transcribe = pd.read_csv(input_path, encoding="ISO-8859-1")

prompt = (
        f"Read the text provided and try and extract the latitude and longitude"
        f"Return the data in JSON format with 'latitude' and 'longitude' as keys"
)


headers = get_headers(my_api_key)

print("####################################### START OUTPUT ######################################")
for index, row in df_to_transcribe.iloc[0:3].iterrows():  

    text_input = row["AI_verbatim"]
    
    
    print(f"\n########################## OCR OUTPUT {text_input[0:10]} ##########################")
    print(f"count: {index}")
    
    payload = make_text_payload(model=MODEL, prompt=prompt, text_input=text_input, num_tokens=4096)

    num_tries = 3
    for i in range(num_tries):
        ocr_output = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        
        response_code = ocr_output.status_code
        if response_code != 200:
            # NOT 200
            print(f"======= 200 not returned {response_code}. Trying request again number: {i} ===========================")
            time.sleep(0.5)
        else:
            # YES 200
            json_returned = clean_up_ocr_output_json_content(ocr_output)
            json_valid = is_json(json_returned)
            if json_valid == False:
                # INVALID JSON
                print(f"======= Returned JSON content not valid. Trying request again number: {i} ===========================")
                print(f"INVALID JSON content****{json_returned}****")
            else:
                # VALID JSON
                """
                # Have to check that the returned JSON keys are correct 
                # Sometimes ChatGPT just doesn't do as its told and changes the key names!
                if are_keys_valid(json_returned, prompt_key_names) == False:
                    # INVALID KEYS
                    print(f"======= Returned JSON contains invalid keys. Trying request again number: {i} ===========================")
                else:
                    # VALID KEYS
                    break
                """
            
                
    """           
                
                
    ###### eo try requests three times

    # OK - we've tried three time to get
    # 1. 200 returned AND
    # 2. valid JSON returned AND
    # 3. valid key names
    # Now we have to create a valid Dict line for the spreadsheet
    error_message = "OK"
    dict_returned = dict()
    if response_code != 200:
        # NOT 200
        # Make a Dict line from the standard empty Dict and 
        # put the whole of the returned message in the OcrText field
        print("RAW ocr_output ****", ocr_output.json(),"****")                   
        dict_returned = eval(str(empty_output_dict))
        dict_returned['MyOcrText'] = str(ocr_output.json())
        error_message = "200 NOT RETURNED FROM GPT"
        print(error_message)
    else:
        # YES 200
        print(f"content****{json_returned}****")
    
        if is_json(json_returned):
            # VALID JSON
            
            # Have to deal with the possibility of invalid keys returned in the valid JSON
            if are_keys_valid(json_returned, prompt_key_names):
                # VALID KEYS
                # Now change all the key names from the human readable used in the prompt to 
                # DataFrame output names to match the NY spreadsheet
                
                dict_returned = eval(json_returned) # JSON -> Dict
                
                for df_name, prompt_name in ocr_column_names:
                    dict_returned[df_name] = dict_returned.pop(prompt_name)
            else:
                # INVALID KEYS
                dict_returned = eval(str(empty_output_dict))
                dict_returned['MyOcrText'] = str(json_returned)                  
                error_message = "INVALID JSON KEYS RETURNED FROM GPT"
                print(error_message)
        else:
            # INVALID JSON
            # Make a Dict line from the standard empty Dict and 
            # just put the invalid JSON in the OcrText field
            dict_returned = eval(str(empty_output_dict))
            dict_returned['MyOcrText'] = str(json_returned)
            error_message = "JSON NOT RETURNED FROM GPT"
            print(error_message)
        
    ###### EO dealing with various types of returned code ######
    
    dict_returned["ERROR"] = str(error_message)  # Insert error message into output
    
    df_to_transcribe.loc[index, dict_returned.keys()] = dict_returned.values() 
    
    if index % batch_size == 0 and index != 0:
        print(f"WRITING BATCH:{index}")
        output_path = f"{output_folder}/{project_name}_{time_stamp}-{index:04}"
        save_dataframe_to_csv(df_to_save=df_to_transcribe, output_path=output_path)
        
    """
   

#################################### eo for loop ####################################

# For safe measure and during testing where batches are not %batch_size
print(f"WRITING BATCH:{index}")
output_path = f"{output_folder}/{project_name}_{time_stamp}-{index:04}"
save_dataframe_to_csv(df_to_save=df_to_transcribe, output_path=output_path)

print("####################################### END OUTPUT ######################################")
  

  


