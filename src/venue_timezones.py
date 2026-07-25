"""IANA time zones for 2026 World Cup host cities (venue-local kickoff times)."""

from datetime import datetime

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

HOST_CITY_TIMEZONE = {
    "Arlington": "America/Chicago",
    "Atlanta": "America/New_York",
    "Boston": "America/New_York",
    "Dallas": "America/Chicago",
    "East Rutherford": "America/New_York",
    "Foxborough": "America/New_York",
    "Guadalajara": "America/Mexico_City",
    "Guadalupe": "America/Monterrey",
    "Houston": "America/Chicago",
    "Inglewood": "America/Los_Angeles",
    "Kansas City": "America/Chicago",
    "Los Angeles": "America/Los_Angeles",
    "Mexico City": "America/Mexico_City",
    "Miami Gardens": "America/New_York",
    "Monterrey": "America/Monterrey",
    "Philadelphia": "America/New_York",
    "San Francisco Bay": "America/Los_Angeles",
    "Santa Clara": "America/Los_Angeles",
    "Seattle": "America/Los_Angeles",
    "Toronto": "America/Toronto",
    "Vancouver": "America/Vancouver",
}


def parse_schedule_datetime(date_str, time_str, host_city):
    """Parse venue-local date/time to UTC ISO string."""
    tz_name = HOST_CITY_TIMEZONE.get(str(host_city).strip(), "UTC")
    month, day, year = [int(x) for x in str(date_str).split("/")]
    parts = str(time_str).split(":")
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0
    dt_local = datetime(year, month, day, hour, minute, tzinfo=ZoneInfo(tz_name))
    return dt_local.isoformat(), tz_name
