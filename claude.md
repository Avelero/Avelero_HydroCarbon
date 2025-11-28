# Bulk Product Generator - Development Guidelines

## Project Overview
This project generates bulk product data for fashion items using Google Gemini API. It consists of two main modules: data creation and data correction.

---

## Project Structure

```
bulk_product_generator/
├── .claude/                      # Claude Code settings (DO NOT COMMIT)
│   └── settings.local.json
│
├── data_creation/                # Product data generation module
│   ├── config/                   # Configuration files
│   │   └── config.py            # Central configuration (API keys, paths, settings)
│   │
│   ├── data/                     # Input data files
│   │   └── categories_rows.json # Product category definitions
│   │
│   ├── docs/                     # Documentation
│   │   └── README.md            # Module-specific documentation
│   │
│   ├── output/                   # Generated output files (DO NOT COMMIT)
│   │   ├── *.csv                # Generated product datasets
│   │   ├── *.json               # Analysis reports
│   │   └── checkpoints/         # Recovery checkpoints
│   │
│   ├── scripts/                  # Executable scripts
│   │   ├── main.py              # Main entry point
│   │   ├── generate_large.py    # Large dataset generation
│   │   └── quick_start.sh       # Quick start bash script
│   │
│   ├── src/                      # Source code modules
│   │   ├── analyzer.py          # Dataset analysis utilities
│   │   ├── checkpoint.py        # Checkpoint/recovery management
│   │   ├── csv_writer.py        # CSV writing utilities
│   │   ├── generator.py         # Core Gemini API generator
│   │   ├── prompts.py           # Prompt templates
│   │   ├── rate_limiter.py      # API rate limiting
│   │   └── vocabularies.py      # Category and country vocabularies
│   │
│   ├── .env.example             # Environment variables template
│   └── requirements.txt         # Python dependencies
│
├── data_correction/              # Product data correction module
│   ├── input/                    # Input files for correction (DO NOT COMMIT)
│   └── .gitkeep                 # Keep empty directory in git
│
├── .env                          # Environment variables (DO NOT COMMIT)
└── .gitignore                   # Git exclusion rules
```

---

## Python Coding Standards

### General Principles
1. **Follow PEP 8**: Python Enhancement Proposal 8 is the style guide for Python code
2. **Write Pythonic Code**: Leverage Python's idioms and features
3. **Readability Counts**: Code is read more often than it is written
4. **Explicit is Better than Implicit**: Clear code over clever code

### File Organization

```python
"""
Module docstring: Brief description of module purpose.

Detailed explanation of what this module does, its main classes,
and how it fits into the overall project architecture.
"""

# Standard library imports
import os
import sys
from typing import List, Dict, Optional

# Third-party imports
import pandas as pd
import numpy as np

# Local application imports
from config import settings
from utils import helper_functions
```

### Naming Conventions

```python
# Constants: UPPER_CASE_WITH_UNDERSCORES
MAX_RETRIES = 3
API_TIMEOUT = 30
DEFAULT_BATCH_SIZE = 100

# Classes: PascalCase
class ProductGenerator:
    pass

class DataValidator:
    pass

# Functions and variables: snake_case
def generate_product_data():
    pass

def validate_input_schema():
    pass

user_count = 0
total_products = 1000

# Private methods/attributes: _leading_underscore
def _internal_helper():
    pass

_cache_data = {}

# Protected in subclasses: __double_leading_underscore
def __reset_state():
    pass
```

### Function Documentation

**Every function MUST have a docstring** explaining its purpose, parameters, return values, and exceptions.

```python
def generate_batch_products(
    category: str,
    count: int,
    quality_level: str = "high"
) -> List[Dict[str, str]]:
    """
    Generate a batch of product data for a specific category.

    This function calls the Gemini API to generate realistic product data
    based on the provided category. It handles retries, rate limiting, and
    validates the generated data before returning.

    Args:
        category: The product category (e.g., "dresses", "shoes", "accessories")
        count: Number of products to generate (1-100)
        quality_level: Quality of generation - "high", "medium", or "low"
                      Higher quality uses more tokens but better results

    Returns:
        List of dictionaries, each containing product fields:
        - name: Product name
        - description: Detailed description
        - price: Price in USD
        - brand: Brand name
        - category: Product category

    Raises:
        ValueError: If category is not in allowed categories
        APIError: If Gemini API fails after max retries
        ValidationError: If generated data fails validation

    Example:
        >>> products = generate_batch_products("dresses", 10, "high")
        >>> len(products)
        10
        >>> products[0].keys()
        dict_keys(['name', 'description', 'price', 'brand', 'category'])
    """
    # Implementation here
    pass
```

### Class Documentation

```python
class CheckpointManager:
    """
    Manages checkpoint creation and recovery for long-running generation tasks.

    This class handles saving progress at regular intervals to prevent data loss
    during large dataset generation. It supports resuming from the last successful
    checkpoint in case of failures or interruptions.

    Attributes:
        checkpoint_dir: Directory where checkpoint files are stored
        interval: Number of products between checkpoints
        current_count: Number of products generated in current session
        last_checkpoint: Timestamp of last checkpoint saved

    Example:
        >>> manager = CheckpointManager(checkpoint_dir="./checkpoints")
        >>> manager.save_checkpoint(products_data, batch_id=5)
        >>> recovered_data = manager.load_latest_checkpoint()
    """

    def __init__(self, checkpoint_dir: str, interval: int = 100):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Path to directory for storing checkpoints
            interval: Number of products between automatic checkpoints

        Raises:
            OSError: If checkpoint directory cannot be created
        """
        self.checkpoint_dir = checkpoint_dir
        self.interval = interval
        # Implementation
```

### Code Comments

Use comments to explain **WHY**, not **WHAT**. The code itself should be clear enough to show what it does.

```python
# GOOD: Explains WHY
# Use exponential backoff to avoid hitting rate limits during peak hours
retry_delay = base_delay * (2 ** attempt)

# Gemini API sometimes returns malformed JSON with trailing commas
# Strip them before parsing to prevent JSONDecodeError
cleaned_response = response.strip().rstrip(',')

# BAD: Explains WHAT (obvious from code)
# Increment counter by 1
counter += 1

# Set name to John
name = "John"
```

### Error Handling

```python
def process_api_response(response: dict) -> dict:
    """
    Process and validate API response from Gemini.

    Args:
        response: Raw response dictionary from API

    Returns:
        Validated and cleaned response data

    Raises:
        ValidationError: If response format is invalid
        DataError: If required fields are missing
    """
    try:
        # Attempt to extract and validate data
        validated_data = validate_response_schema(response)

    except KeyError as e:
        # Log the specific missing key for debugging
        logger.error(f"Missing required field in API response: {e}")
        raise DataError(f"API response missing field: {e}") from e

    except json.JSONDecodeError as e:
        # API sometimes returns malformed JSON, log for investigation
        logger.error(f"Malformed JSON in API response: {response[:100]}")
        raise ValidationError("Could not parse API response as JSON") from e

    else:
        # Only reached if try block succeeds with no exceptions
        logger.info(f"Successfully validated response with {len(validated_data)} records")
        return validated_data

    finally:
        # Always executed, regardless of exceptions
        # Release any resources (file handles, connections, etc.)
        cleanup_resources()
```

### Type Hints

Use type hints for better code clarity and IDE support:

```python
from typing import List, Dict, Optional, Union, Tuple, Any

def analyze_products(
    products: List[Dict[str, Any]],
    filters: Optional[Dict[str, str]] = None,
    limit: int = 100
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Analyze product dataset and return statistics.

    Args:
        products: List of product dictionaries
        filters: Optional filters to apply before analysis
        limit: Maximum number of products to analyze

    Returns:
        Tuple containing:
        - DataFrame with analyzed products
        - Dictionary with statistics (counts, averages, etc.)
    """
    pass
```

### Python Best Practices

```python
# Use list comprehensions for simple transformations
# GOOD
product_names = [p['name'] for p in products if p['price'] > 50]

# AVOID
product_names = []
for p in products:
    if p['price'] > 50:
        product_names.append(p['name'])


# Use context managers for resource management
# GOOD
with open('products.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(data)

# AVOID
f = open('products.csv', 'w')
writer = csv.writer(f)
writer.writerows(data)
f.close()  # Might not execute if exception occurs


# Use f-strings for string formatting (Python 3.6+)
# GOOD
message = f"Generated {count} products in {duration:.2f} seconds"

# AVOID
message = "Generated {} products in {:.2f} seconds".format(count, duration)
message = "Generated " + str(count) + " products in " + str(duration) + " seconds"


# Use enumerate instead of manual counter
# GOOD
for index, product in enumerate(products):
    print(f"Product {index}: {product['name']}")

# AVOID
index = 0
for product in products:
    print(f"Product {index}: {product['name']}")
    index += 1


# Use dict.get() for optional keys
# GOOD
brand = product.get('brand', 'Unknown')
description = product.get('description', '')

# AVOID
brand = product['brand'] if 'brand' in product else 'Unknown'
```

---

## C Coding Standards

### General Principles
1. **Follow MISRA C Guidelines**: Industry standard for reliable C code
2. **Defensive Programming**: Always validate inputs and check return values
3. **Memory Safety**: Prevent buffer overflows, memory leaks, and dangling pointers
4. **Portability**: Write platform-independent code where possible

### File Organization

```c
/**
 * @file product_generator.c
 * @brief Product data generator using external API
 *
 * This module handles communication with the Gemini API to generate
 * realistic product data. It manages memory allocation, API requests,
 * response parsing, and error handling.
 *
 * @author Your Name
 * @date 2025-11-28
 * @version 1.0
 */

/* Standard library includes */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

/* Third-party library includes */
#include <curl/curl.h>
#include <json-c/json.h>

/* Project includes */
#include "config.h"
#include "product_generator.h"
#include "error_handler.h"

/* Constants and macros */
#define MAX_PRODUCT_NAME_LEN 256
#define API_TIMEOUT_SECONDS 30
#define MAX_RETRY_ATTEMPTS 3

/* Type definitions */
typedef struct {
    char name[MAX_PRODUCT_NAME_LEN];
    char description[1024];
    double price;
    char brand[128];
} Product;

/* Static (private) function declarations */
static int validate_product_data(const Product *product);
static char* build_api_request(const char *category, int count);
static void cleanup_resources(void);

/* Global variables (avoid when possible, use static if needed) */
static CURL *g_curl_handle = NULL;
static int g_total_products_generated = 0;
```

### Naming Conventions

```c
/* Constants and macros: UPPER_CASE_WITH_UNDERSCORES */
#define MAX_BUFFER_SIZE 4096
#define DEFAULT_TIMEOUT 30
#define API_BASE_URL "https://api.example.com"

/* Functions: snake_case */
int generate_product_batch(const char *category, int count);
bool validate_api_response(const char *response);
void cleanup_generator(void);

/* Global variables: g_ prefix (avoid when possible) */
static int g_error_count = 0;
static char g_api_key[256];

/* Static variables: s_ prefix */
static int s_request_counter = 0;
static bool s_initialized = false;

/* Structs and types: PascalCase with _t suffix */
typedef struct ProductData_t {
    char *name;
    double price;
    int stock_count;
} ProductData_t;

/* Enums: PascalCase */
typedef enum {
    STATUS_SUCCESS = 0,
    STATUS_ERROR = -1,
    STATUS_RETRY = 1
} StatusCode;
```

### Function Documentation

**Every function MUST have a detailed comment block** explaining its purpose, parameters, return values, and side effects.

```c
/**
 * @brief Generate a batch of product data for specified category
 *
 * This function communicates with the Gemini API to generate realistic
 * product data. It handles memory allocation for the returned products,
 * performs input validation, and implements retry logic for transient
 * API failures.
 *
 * Memory Management:
 * - Allocates memory for product array (caller must free)
 * - Each product string is dynamically allocated (caller must free)
 *
 * @param[in] category Product category (must be null-terminated, max 64 chars)
 * @param[in] count Number of products to generate (1-100)
 * @param[out] products Pointer to array of products (allocated by function)
 * @param[out] error_msg Buffer for error message (min 256 bytes, can be NULL)
 *
 * @return Number of products generated on success, -1 on error
 *
 * @retval >0 Success, number of products generated
 * @retval -1 Invalid input parameters
 * @retval -2 API communication error
 * @retval -3 Memory allocation failure
 * @retval -4 Response parsing error
 *
 * @note Caller is responsible for freeing allocated memory using free_products()
 * @warning This function is not thread-safe due to global CURL handle
 *
 * @see free_products(), validate_category()
 *
 * Example:
 * @code
 *   Product *products = NULL;
 *   char error[256];
 *   int count = generate_product_batch("dresses", 10, &products, error);
 *   if (count > 0) {
 *       // Use products
 *       free_products(products, count);
 *   } else {
 *       fprintf(stderr, "Error: %s\n", error);
 *   }
 * @endcode
 */
int generate_product_batch(
    const char *category,
    int count,
    Product **products,
    char *error_msg
) {
    /* Input validation */
    if (category == NULL || products == NULL) {
        if (error_msg != NULL) {
            snprintf(error_msg, 256, "Invalid NULL parameter");
        }
        return -1;
    }

    /* Implementation */
}
```

### Code Comments

```c
/**
 * @brief Initialize the product generator with API credentials
 *
 * This function must be called before any generation operations.
 * It sets up the CURL handle and validates the API key.
 *
 * @param[in] api_key API key for Gemini (must be 40 chars hex string)
 * @return true on success, false on failure
 */
bool init_generator(const char *api_key) {
    /* Validate API key format before making any requests */
    if (api_key == NULL || strlen(api_key) != 40) {
        fprintf(stderr, "Invalid API key format\n");
        return false;
    }

    /* Initialize CURL handle */
    /* Using global handle to reuse connections for better performance */
    g_curl_handle = curl_easy_init();
    if (g_curl_handle == NULL) {
        fprintf(stderr, "Failed to initialize CURL\n");
        return false;
    }

    /* Set timeout to prevent indefinite hangs on network issues */
    curl_easy_setopt(g_curl_handle, CURLOPT_TIMEOUT, API_TIMEOUT_SECONDS);

    /* Enable follow redirects, API endpoint may redirect to CDN */
    curl_easy_setopt(g_curl_handle, CURLOPT_FOLLOWLOCATION, 1L);

    s_initialized = true;
    return true;
}
```

### Memory Management

```c
/**
 * @brief Allocate and initialize a new product structure
 *
 * @return Pointer to allocated product, or NULL on failure
 * @note Caller must free using free_product()
 */
Product* create_product(void) {
    /* Allocate memory and zero-initialize for safety */
    Product *product = (Product*)calloc(1, sizeof(Product));
    if (product == NULL) {
        fprintf(stderr, "Memory allocation failed for product\n");
        return NULL;
    }

    /* Initialize with safe defaults */
    product->price = 0.0;
    product->stock_count = 0;
    product->name = NULL;

    return product;
}

/**
 * @brief Free product and all associated dynamic memory
 *
 * @param[in,out] product Product to free (set to NULL after freeing)
 */
void free_product(Product **product) {
    if (product == NULL || *product == NULL) {
        return;  /* Safe to call with NULL */
    }

    /* Free dynamically allocated strings */
    if ((*product)->name != NULL) {
        free((*product)->name);
        (*product)->name = NULL;
    }

    /* Free the product structure itself */
    free(*product);
    *product = NULL;  /* Prevent use-after-free bugs */
}
```

### Error Handling

```c
/**
 * @brief Parse JSON response from API into product structure
 *
 * @param[in] json_str JSON string from API response
 * @param[out] product Pointer to product structure to fill
 * @return Status code indicating success or specific error
 */
StatusCode parse_api_response(const char *json_str, Product *product) {
    json_object *root = NULL;
    json_object *name_obj = NULL;
    const char *name_str = NULL;

    /* Validate inputs before proceeding */
    if (json_str == NULL || product == NULL) {
        return STATUS_ERROR;
    }

    /* Parse JSON string */
    root = json_tokener_parse(json_str);
    if (root == NULL) {
        fprintf(stderr, "Failed to parse JSON response\n");
        return STATUS_ERROR;
    }

    /* Extract name field with error checking */
    if (!json_object_object_get_ex(root, "name", &name_obj)) {
        fprintf(stderr, "Missing 'name' field in response\n");
        json_object_put(root);  /* Always clean up before returning */
        return STATUS_ERROR;
    }

    /* Get string value and validate */
    name_str = json_object_get_string(name_obj);
    if (name_str == NULL || strlen(name_str) == 0) {
        fprintf(stderr, "Invalid or empty product name\n");
        json_object_put(root);
        return STATUS_ERROR;
    }

    /* Allocate and copy name (with bounds checking) */
    size_t name_len = strlen(name_str);
    if (name_len >= MAX_PRODUCT_NAME_LEN) {
        fprintf(stderr, "Product name too long: %zu bytes\n", name_len);
        json_object_put(root);
        return STATUS_ERROR;
    }

    /* Use strncpy with explicit null termination for safety */
    strncpy(product->name, name_str, MAX_PRODUCT_NAME_LEN - 1);
    product->name[MAX_PRODUCT_NAME_LEN - 1] = '\0';

    /* Clean up JSON object */
    json_object_put(root);

    return STATUS_SUCCESS;
}
```

### C Best Practices

```c
/* Always check malloc/calloc return values */
/* GOOD */
char *buffer = (char*)malloc(size);
if (buffer == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    return -1;
}

/* AVOID - never use malloc without checking */
char *buffer = (char*)malloc(size);
strcpy(buffer, data);  /* Could crash if malloc failed */


/* Use const for pointers that won't modify data */
/* GOOD */
int calculate_total(const Product *products, int count);
void print_name(const char *name);

/* Prevents accidental modifications */


/* Always null-terminate strings */
/* GOOD */
strncpy(dest, src, MAX_LEN - 1);
dest[MAX_LEN - 1] = '\0';  /* Ensure null termination */

/* AVOID */
strncpy(dest, src, MAX_LEN);  /* May not be null-terminated */


/* Check array bounds before access */
/* GOOD */
if (index >= 0 && index < array_size) {
    value = array[index];
}

/* AVOID */
value = array[index];  /* Could be out of bounds */


/* Initialize variables at declaration */
/* GOOD */
int count = 0;
char *buffer = NULL;
Product *product = NULL;

/* AVOID */
int count;
char *buffer;
Product *product;
/* ... later usage without initialization */


/* Use sizeof for buffer sizes, not hardcoded numbers */
/* GOOD */
char name[MAX_NAME_LEN];
memset(name, 0, sizeof(name));
snprintf(name, sizeof(name), "Product %d", id);

/* AVOID */
memset(name, 0, 256);  /* Magic number, error-prone */
snprintf(name, 256, "Product %d", id);
```

---

## Professional Commenting Guidelines

### When to Comment

✅ **DO comment:**
- Complex algorithms or business logic
- Non-obvious optimizations
- Workarounds for bugs in external libraries
- Security-sensitive code sections
- Performance-critical sections
- Public API functions (always)
- Assumptions and constraints
- Known limitations or TODOs

❌ **DON'T comment:**
- Obvious code that explains itself
- Every single line (code should be self-documenting)
- Commented-out code (use version control instead)
- Redundant information already in function name

### Comment Quality Examples

```python
# ✅ EXCELLENT: Explains WHY and provides context
# Use binary search instead of linear scan for large datasets
# Performance testing showed 100x improvement for 10k+ products
# Trade-off: requires sorted data (sorted once during initialization)
def find_product_by_id(product_id: int) -> Optional[Product]:
    return binary_search(self.sorted_products, product_id)

# ❌ BAD: States the obvious
# Search for product
def find_product_by_id(product_id: int) -> Optional[Product]:
    return binary_search(self.sorted_products, product_id)


# ✅ EXCELLENT: Explains workaround and references source
# Gemini API occasionally returns prices with commas (e.g., "1,299.99")
# This is not documented but observed in production
# Strip commas before conversion to float
# See: https://github.com/google/gemini/issues/1234
price_str = response['price'].replace(',', '')
price = float(price_str)

# ❌ BAD: Obvious what the code does
# Remove commas from price string
price_str = response['price'].replace(',', '')


# ✅ EXCELLENT: Documents security consideration
# Sanitize user input to prevent SQL injection attacks
# Even though we use parameterized queries, defense in depth principle
# Only allow alphanumeric characters and basic punctuation
sanitized_input = re.sub(r'[^a-zA-Z0-9\s\-_.,]', '', user_input)

# ❌ BAD: Doesn't explain the "why"
# Clean the input
sanitized_input = re.sub(r'[^a-zA-Z0-9\s\-_.,]', '', user_input)
```

---

## Git Workflow

### Commit Messages

Use clear, descriptive commit messages:

```
Format: <type>: <short summary>

<detailed description if needed>

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- refactor: Code refactoring
- test: Adding tests
- chore: Build process, dependencies, etc.

Example:
feat: Add checkpoint recovery for large dataset generation

Implemented checkpoint system that saves progress every 100 products.
This prevents data loss during long-running generation tasks.
Includes automatic recovery from last checkpoint on restart.
```

### Branching Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: Individual features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Urgent production fixes

---

## Testing Requirements

### Python Testing

```python
def test_generate_product_batch():
    """
    Test that batch generation creates correct number of products.

    Verifies:
    - Correct count of products returned
    - Each product has required fields
    - Price values are positive numbers
    """
    generator = GeminiGenerator()
    products = generator.generate_batch("dresses", count=5)

    assert len(products) == 5
    for product in products:
        assert 'name' in product
        assert 'price' in product
        assert float(product['price']) > 0
```

### C Testing

```c
/**
 * @brief Test product creation and memory cleanup
 *
 * Verifies:
 * - Product allocation succeeds
 * - Fields are initialized to safe defaults
 * - Free operation prevents memory leaks
 */
void test_create_and_free_product(void) {
    Product *product = create_product();
    assert(product != NULL);
    assert(product->price == 0.0);
    assert(product->stock_count == 0);

    free_product(&product);
    assert(product == NULL);  /* Verify set to NULL after free */
}
```

---

## Security Considerations

1. **Never commit sensitive data**: API keys, passwords, tokens
2. **Validate all external input**: User input, API responses, file contents
3. **Use environment variables**: For configuration and secrets
4. **Sanitize file paths**: Prevent directory traversal attacks
5. **Check buffer sizes**: Prevent buffer overflow in C code
6. **Use parameterized queries**: Prevent SQL injection
7. **Update dependencies**: Regular security patches

---

## Code Review Checklist

Before committing code, verify:

- [ ] All functions have proper docstrings/comments
- [ ] Error handling is comprehensive
- [ ] Memory is properly managed (no leaks)
- [ ] Input validation is performed
- [ ] Type hints are used (Python)
- [ ] Code follows naming conventions
- [ ] Tests are included for new features
- [ ] No sensitive data in commits
- [ ] Code is formatted consistently
- [ ] No compiler warnings (C)
- [ ] No linter warnings (Python)

---

## Development Environment Setup

### Python
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
cd data_creation
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### C
```bash
# Install required libraries
sudo apt-get install libcurl4-openssl-dev libjson-c-dev  # Ubuntu/Debian
brew install curl json-c                                  # macOS

# Compile with warnings enabled
gcc -Wall -Wextra -Werror -pedantic -std=c11 -o program main.c
```

---

## Additional Resources

- [PEP 8 - Python Style Guide](https://peps.python.org/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [MISRA C Guidelines](https://www.misra.org.uk/)
- [Linux Kernel Coding Style](https://www.kernel.org/doc/html/latest/process/coding-style.html)
- [Semantic Versioning](https://semver.org/)

---

**Last Updated**: 2025-11-28
**Version**: 1.0
