import requests
from mcp.server.fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("Currency Converter")


@mcp.tool()
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Convert an amount from one currency to another using Frankfurter v2 API.

    Args:
        amount: Amount to convert
        from_currency: Source currency (e.g. USD)
        to_currency: Target currency (e.g. EGP)

    Returns:
        Formatted conversion result string
    """

    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

    # ✅ Best v2 endpoint (simple + direct)
    url = f"https://api.frankfurter.dev/v2/rate/{from_currency}/{to_currency}"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        data = response.json()

        # Extract rate
        rate = data.get("rate")

        if rate is None:
            return f"Could not find exchange rate for {from_currency} → {to_currency}"

        converted_amount = amount * rate

        return (
            f"{amount} {from_currency} = {converted_amount:.2f} {to_currency} "
            f"(Rate: {rate})"
        )

    except requests.exceptions.RequestException as e:
        return f"Error converting currency: {str(e)}"


# -------------------------
# Test call
# -------------------------
if __name__ == "__main__":
    print("Testing Currency Converter:")
    result = convert_currency(500, "USD", "EGP")
    print(result)