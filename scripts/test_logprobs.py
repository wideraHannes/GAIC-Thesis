#!/usr/bin/env python3
"""
Test if Mistral API supports logprobs for confidence extraction.
"""

import os
import math
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

messages = [
    {"role": "system", "content": "Classify as 'Argument' or 'No-Argument'. Respond with only the label."},
    {"role": "user", "content": "Climate change is the biggest threat to humanity."},
]

print("=== Testing Mistral API for logprobs ===\n")

# Check available parameters in chat.complete
import inspect
sig = inspect.signature(client.chat.complete)
print(f"Available parameters: {list(sig.parameters.keys())}\n")

# Try basic call first
response = client.chat.complete(
    model="mistral-small-latest",
    messages=messages,
    temperature=0,
    max_tokens=10,
)

print(f"Response: {response.choices[0].message.content}")
print(f"\nChoice attributes: {[a for a in dir(response.choices[0]) if not a.startswith('_')]}")

if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
    print(f"\n✓ Logprobs found: {response.choices[0].logprobs}")
else:
    print(f"\n✗ No logprobs in basic response")

# Try with explicit logprobs parameter if supported
print("\n" + "="*50)
print("Trying with logprobs parameter...\n")

try:
    response2 = client.chat.complete(
        model="mistral-small-latest",
        messages=messages,
        temperature=0,
        max_tokens=10,
        logprobs=True,
    )

    print(f"Response: {response2.choices[0].message.content}")

    if hasattr(response2.choices[0], 'logprobs') and response2.choices[0].logprobs:
        print(f"\n✓ Logprobs enabled!")
        logprobs = response2.choices[0].logprobs
        print(f"Logprobs object: {logprobs}")

        # Try to extract token probabilities
        if hasattr(logprobs, 'content') and logprobs.content:
            for token_info in logprobs.content:
                prob = math.exp(token_info.logprob)
                print(f"  Token: '{token_info.token}' - prob: {prob:.4f}")
                if hasattr(token_info, 'top_logprobs') and token_info.top_logprobs:
                    print("  Alternatives:")
                    for alt in token_info.top_logprobs:
                        alt_prob = math.exp(alt.logprob)
                        print(f"    '{alt.token}': {alt_prob:.4f}")
    else:
        print("✗ logprobs parameter accepted but no data returned")

except TypeError as e:
    print(f"✗ logprobs parameter not supported: {e}")
except Exception as e:
    print(f"Error: {e}")

# Try OpenAI-compatible endpoint
print("\n" + "="*50)
print("Testing via OpenAI-compatible endpoint...\n")

from openai import OpenAI

openai_client = OpenAI(
    api_key=os.getenv("MISTRAL_API_KEY"),
    base_url="https://api.mistral.ai/v1",
)

try:
    response3 = openai_client.chat.completions.create(
        model="mistral-small-latest",
        messages=messages,  # type: ignore
        temperature=0,
        max_tokens=10,
        logprobs=True,
        top_logprobs=5,
    )

    print(f"Response: {response3.choices[0].message.content}")

    if response3.choices[0].logprobs:
        print(f"\n✓ Logprobs work via OpenAI endpoint!")
        for token_info in response3.choices[0].logprobs.content or []:
            prob = math.exp(token_info.logprob)
            print(f"  Token: '{token_info.token}' - prob: {prob:.4f}")
            if token_info.top_logprobs:
                print("  Top alternatives:")
                for alt in token_info.top_logprobs[:3]:
                    alt_prob = math.exp(alt.logprob)
                    print(f"    '{alt.token}': {alt_prob:.4f}")
    else:
        print("✗ No logprobs via OpenAI endpoint either")

except Exception as e:
    print(f"Error with OpenAI endpoint: {e}")

# Try Azure/Portkey
print("\n" + "="*50)
print("Testing via Portkey (Azure GPT-4.1)...\n")

from portkey_ai import PORTKEY_GATEWAY_URL

portkey_client = OpenAI(
    api_key=os.getenv("PORTKEY_API_KEY"),
    base_url=PORTKEY_GATEWAY_URL,
)

try:
    response4 = portkey_client.chat.completions.create(
        model="@azure-openai-foundry/gpt-4.1",
        messages=messages,  # type: ignore
        temperature=0,
        max_tokens=10,
        logprobs=True,
        top_logprobs=5,
    )

    print(f"Response: {response4.choices[0].message.content}")

    if response4.choices[0].logprobs:
        print(f"\n✓ Logprobs work via Portkey/Azure!")
        for token_info in response4.choices[0].logprobs.content or []:
            prob = math.exp(token_info.logprob)
            print(f"  Token: '{token_info.token}' - prob: {prob:.4f}")
            if token_info.top_logprobs:
                print("  Top alternatives:")
                for alt in token_info.top_logprobs[:3]:
                    alt_prob = math.exp(alt.logprob)
                    print(f"    '{alt.token}': {alt_prob:.4f}")
    else:
        print("✗ No logprobs via Portkey either")

except Exception as e:
    print(f"Error with Portkey: {e}")
