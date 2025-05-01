import pandas as pd
import random
from faker import Faker

# Initialize Faker for realistic data
fake = Faker()

# Define query templates and categories
query_templates = {
    "price_inquiry": [
        "What's the cheapest flight to {destination}?",
        "How much is a ticket to {destination} on {date}?",
        "Flight prices to {destination} for {date}?"
    ],
    "booking_change": [
        "Can I change my flight to {destination}?",
        "How do I reschedule my booking to {destination} for {date}?",
        "Change my flight from {origin} to {destination}"
    ],
    "cancellation": [
        "Cancel my booking to {destination}",
        "How do I cancel my flight on {date}?",
        "Is there a fee to cancel my ticket to {destination}?"
    ],
    "flight_status": [
        "What's the status of my flight to {destination}?",
        "Is the flight to {destination} on {date} on time?",
        "Check flight status for {destination}"
    ],
    "baggage": [
        "What's the baggage allowance for my flight to {destination}?",
        "Can I add extra luggage to my {destination} booking?",
        "Baggage rules for {destination} flight"
    ],
    "loyalty": [
        "How many miles do I earn for a flight to {destination}?",
        "Can I use my loyalty points for a ticket to {destination}?",
        "Check my loyalty status for {destination} flights"
    ],
    "refunds": [
        "Can I get a refund for my {destination} ticket?",
        "How do I process a refund for a flight on {date}?",
        "Refund policy for {destination} bookings"
    ],
    "check_in": [
        "How do I check in for my flight to {destination}?",
        "Online check-in for {destination} flight on {date}",
        "Can I check in early for {destination}?"
    ],
    "special_requests": [
        "Can I request a vegetarian meal for my {destination} flight?",
        "Wheelchair assistance for my flight to {destination}",
        "Special seating for {destination} booking"
    ],
    "other": [
        "What’s the weather like in {destination}?",
        "Contact details for your airline",
        "What are your COVID policies?"
    ]
}

# Destinations and dates for variety
destinations = ["Lagos", "Abuja", "Accra", "London", "New York", "Paris"]
origins = ["Lagos", "Abuja", "Accra"]
dates = [fake.date_between(start_date="today", end_date="+30d").strftime("%Y-%m-%d") for _ in range(20)]

# Generate queries
data = []
for _ in range(1000):
    category = random.choice(list(query_templates.keys()))
    template = random.choice(query_templates[category])
    destination = random.choice(destinations)
    origin = random.choice(origins)
    date = random.choice(dates)
    # Format query with placeholders
    query = template.format(destination=destination, origin=origin, date=date)
    # Add slight variations (e.g., typos, rephrasing)
    if random.random() < 0.2:  # 20% chance of variation
        query = query.replace("flight", "flght") if random.random() < 0.5 else f"Please, {query.lower()}"
    data.append({
        "query": query,
        "category": category,
        "destination": destination,
        "origin": origin,
        "date": date
    })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("queries.csv", index=False)
print(f"Created queries.csv with {len(df)} samples across {len(query_templates)} categories")