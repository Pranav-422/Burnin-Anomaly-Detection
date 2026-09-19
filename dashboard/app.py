"""
[DEPRECATED / ARCHIVED] Legacy Streamlit prototype.

The ASTROLAB production system has migrated to the modern ISRO-themed HTML5 / Tailwind CSS /
Vanilla JS web portal located in `dashboard/static/` (Home, Login, About Us, and Screening Console),
served directly via FastAPI on port 8000 or hosted on Vercel.
"""

if __name__ == "__main__":
    print("Notice: The Streamlit dashboard has been superseded by the ASTROLAB Web Portal.")
    print("To launch the ASTROLAB web portal, run: uvicorn api.main:app --reload --port 8000")
