import os 
import requests
import datetime

# 🔐 Your OpenAI API Key
API_KEY = "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"  # Make sure to set this environment variable
# 📅 from november 2025 
end_date = datetime.date(2025, 11, 30)  # Set the end date to the last day of November 2025
start_date = end_date.replace(day=1)

url = "https://api.openai.com/v1/organization/costs"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

params = {
    "start_date": start_date.isoformat(),
    "end_date": end_date.isoformat()
}

response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    data = response.json()
    
    total_usage = 0
    for item in data.get("data", []):
        total_usage += item.get("cost", 0)

    print(f"Usage from {start_date} to {end_date}: ${total_usage:.4f}")

    # 👉 Set your budget manually here
    budget_limit = 1.0  # dollars
    remaining = budget_limit - total_usage

    print(f"Budget limit: ${budget_limit}")
    print(f"Remaining: ${remaining:.4f}")

else:
    print("Error:", response.status_code, response.text)