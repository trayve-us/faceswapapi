#!/usr/bin/env python3
"""
Startup script for DigitalOcean App Platform
Handles PORT environment variable properly
"""
import os
import sys

def main():
    # Get port from environment variable, default to 8000
    port = int(os.environ.get('PORT', 8000))
    
    print(f"Starting server on port {port}")
    
    # Import uvicorn and the app
    import uvicorn
    from main import app
    
    # Run the server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main()
