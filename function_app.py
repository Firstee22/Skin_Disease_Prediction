"""
Azure Functions entry point.

Azure Functions (Python v2 programming model) looks for a file named
function_app.py. Here we wrap our FastAPI app so that every HTTP request
sent to the Function App is forwarded to FastAPI through ASGI.
"""

import azure.functions as func

from main import app as fastapi_app  # the FastAPI instance defined in main.py

# ANONYMOUS = no function key needed. Use AuthLevel.FUNCTION if you want callers
# to supply a function key (?code=... or x-functions-key header).
app = func.AsgiFunctionApp(
    app=fastapi_app,
    http_auth_level=func.AuthLevel.ANONYMOUS,
)
