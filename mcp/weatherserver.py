from mcp.server.fastmcp import FastMCP

mcp = FastMCP()

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get the weather of the loacation"""
    return "It is rainy in Los Angeles"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")