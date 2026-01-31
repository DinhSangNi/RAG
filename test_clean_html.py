#!/usr/bin/env python3
"""
Test script for clean_wikipedia_html function
"""

import os
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

try:
    from workers.process_worker import clean_wikipedia_html
    print("✅ Import successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_clean_html():
    """Test the clean_wikipedia_html function"""
    test_file = "test_html.html"

    if not os.path.exists(test_file):
        print(f"❌ Test file {test_file} not found")
        return False

    try:
        print(f"🧹 Testing clean_wikipedia_html with {test_file}")
        result_path = clean_wikipedia_html(test_file)
        print(f"✅ Function completed successfully")
        print(f"📁 Output: {result_path}")

        # Check if output file exists
        if os.path.exists(result_path):
            print("✅ Output file created")
            # Read first few lines
            with open(result_path, 'r', encoding='utf-8') as f:
                content = f.read()[:500]
            print(f"📄 Content preview:\n{content}...")
            return True
        else:
            print("❌ Output file not created")
            return False

    except Exception as e:
        print(f"❌ Function failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("🧪 TESTING CLEAN_WIKIPEDIA_HTML FUNCTION")
    print("=" * 50)

    success = test_clean_html()

    print("=" * 50)
    if success:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ TESTS FAILED")
    print("=" * 50)