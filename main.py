from fastmcp import FastMCP

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("mcp-linkedin")
logger = logging.getLogger(__name__)

API_HOST = "linkedin-data-api.p.rapidapi.com"
API_BASE_URL = f"https://{API_HOST}"
SEARCH_JOBS_PAGE_SIZE = 25
SEARCH_JOBS_MAX_START = 975


def get_client() -> httpx.Client:
    """Return an httpx client configured for the LinkedIn Data API."""
    headers = {
        "x-rapidapi-key": os.getenv("RAPIDAPI_KEY"),
        "x-rapidapi-host": API_HOST,
    }
    return httpx.Client(headers=headers, timeout=httpx.Timeout(30.0, connect=10.0))


def _parse_api_response(response: httpx.Response) -> Dict[str, Any]:
    """Raise useful errors for bad HTTP/API responses and return parsed JSON."""
    response.raise_for_status()

    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Unexpected API response shape: expected a JSON object.")

    if not data.get("success", False):
        raise ValueError(f"API Error: {data.get('message', 'Unknown error')}")

    return data


def _extract_location_id(data: Dict[str, Any], keyword: str) -> str:
    items = data.get("data", {}).get("items", [])
    if not items:
        raise ValueError(f"No locations found matching '{keyword}'")

    first_location = items[0]
    full_id = first_location.get("id", "") or ""
    return full_id.split(":")[-1] if ":" in full_id else full_id


def _lookup_location_id(client: httpx.Client, keyword: str) -> str:
    url = f"{API_BASE_URL}/search-locations"
    response = client.get(url, params={"keyword": keyword})
    data = _parse_api_response(response)
    return _extract_location_id(data, keyword)


def _build_job_key(job: Dict[str, Any]) -> str:
    """Build a stable dedupe key for a job record."""
    for field in ("id", "referenceId", "url"):
        value = job.get(field)
        if value:
            return f"{field}:{str(value).strip().lower()}"

    company = job.get("company", {})
    company_name = company.get("name", "") if isinstance(company, dict) else str(company or "")
    title = str(job.get("title", "")).strip().lower()
    location = str(job.get("location", "")).strip().lower()
    post_at = str(job.get("postAt", "")).strip().lower()

    return f"fallback:{title}|{company_name.strip().lower()}|{location}|{post_at}"


_RELATIVE_TIME_RE = re.compile(
    r"(?P<count>\d+)\s*\+?\s*(?P<unit>"
    r"minute|minutes|min|mins|m|"
    r"hour|hours|hr|hrs|h|"
    r"day|days|d|"
    r"week|weeks|w|"
    r"month|months|mo|"
    r"year|years|yr|yrs|y"
    r")\s*ago?",
    re.IGNORECASE,
)


def _parse_posted_at(value: Any) -> Optional[datetime]:
    """
    Parse the API's postAt field into a timezone-aware datetime.

    Supports:
    - Unix timestamps (seconds or milliseconds)
    - ISO-8601 strings
    - Common date strings
    - Relative text like "2 days ago", "3w ago", "30+ days ago"
    """
    if value in (None, "", "Unknown Date"):
        return None

    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 1_000_000_000_000:
            timestamp /= 1000.0
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)

    text = str(value).strip()
    if not text:
        return None

    normalized = text.lower().replace("reposted", "").replace("posted", "").strip()
    normalized = " ".join(normalized.split())

    if normalized in {"just now", "now", "today"}:
        return datetime.now(timezone.utc)
    if normalized == "yesterday":
        return datetime.now(timezone.utc) - timedelta(days=1)

    digits_only = normalized.replace("+", "")
    if digits_only.isdigit():
        timestamp = float(digits_only)
        if timestamp > 1_000_000_000_000:
            timestamp /= 1000.0
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)

    iso_candidate = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_candidate)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    match = _RELATIVE_TIME_RE.search(normalized)
    if not match:
        compact_match = re.search(r"(?P<count>\d+)(?P<unit>mo|w|d|h|m|y)$", normalized, re.IGNORECASE)
        match = compact_match

    if match:
        count = int(match.group("count"))
        unit = match.group("unit").lower()
        now = datetime.now(timezone.utc)

        if unit in {"minute", "minutes", "min", "mins", "m"}:
            return now - timedelta(minutes=count)
        if unit in {"hour", "hours", "hr", "hrs", "h"}:
            return now - timedelta(hours=count)
        if unit in {"day", "days", "d"}:
            return now - timedelta(days=count)
        if unit in {"week", "weeks", "w"}:
            return now - timedelta(weeks=count)
        if unit in {"month", "months", "mo"}:
            return now - timedelta(days=30 * count)
        if unit in {"year", "years", "yr", "yrs", "y"}:
            return now - timedelta(days=365 * count)

    return None


def _normalize_job(job: Dict[str, Any], parsed_posted_at: Optional[datetime]) -> Dict[str, Any]:
    company = job.get("company", {})
    company_name = company.get("name", "Unknown Company") if isinstance(company, dict) else str(company or "Unknown Company")
    company_logo = company.get("logo") if isinstance(company, dict) else None

    return {
        "id": job.get("id"),
        "title": job.get("title", "Unknown Title"),
        "company": company_name,
        "company_logo": company_logo,
        "location": job.get("location", "Unknown Location"),
        "url": job.get("url", ""),
        "post_date": job.get("postAt", "Unknown Date"),
        "post_date_iso": parsed_posted_at.isoformat() if parsed_posted_at else None,
        "reference_id": job.get("referenceId"),
    }


@mcp.tool()
def search_jobs(
    keywords: str,
    limit: int = 10,
    location: str = "United States",
    format_output: bool = True,
    max_age_days: int = 60,
    onsite_remote: Optional[str] = None,
) -> dict:
    """
    Search LinkedIn jobs with pagination, 60-day filtering, and deduplication.

    :param keywords: Job search keywords
    :param limit: Maximum number of job results to return
    :param location: Location filter
    :param format_output: Kept for backward compatibility
    :param max_age_days: Only return jobs posted within the last N days
    :param onsite_remote: Optional LinkedIn remote filter (onSite, remote, hybrid)
    :return: Dictionary with query metadata and deduplicated job listings
    """
    if limit <= 0:
        return {"query": {"keywords": keywords, "location": location}, "count": 0, "jobs": []}

    cutoff = None
    if max_age_days and max_age_days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    try:
        with get_client() as client:
            location_id = _lookup_location_id(client, location)

            jobs: List[Dict[str, Any]] = []
            seen_keys = set()
            start = 0
            raw_seen = 0
            duplicates_removed = 0
            age_filtered_out = 0
            pages_fetched = 0

            while start <= SEARCH_JOBS_MAX_START and len(jobs) < limit:
                params = {
                    "keywords": keywords,
                    "locationId": location_id,
                    "datePosted": "anyTime" if cutoff else "pastMonth",
                    "sort": "mostRecent" if cutoff else "mostRelevant",
                    "start": start,
                }
                if onsite_remote:
                    params["onsiteRemote"] = onsite_remote

                url = f"{API_BASE_URL}/search-jobs?{urlencode(params)}"
                response = client.get(url)
                data = _parse_api_response(response)

                batch = data.get("data", []) or []
                pages_fetched += 1

                if not batch:
                    break

                raw_seen += len(batch)
                parseable_dates: List[datetime] = []

                for job in batch:
                    job_key = _build_job_key(job)
                    if job_key in seen_keys:
                        duplicates_removed += 1
                        continue
                    seen_keys.add(job_key)

                    posted_at_dt = _parse_posted_at(job.get("postAt"))
                    if posted_at_dt is not None:
                        parseable_dates.append(posted_at_dt)

                    if cutoff is not None:
                        if posted_at_dt is None or posted_at_dt < cutoff:
                            age_filtered_out += 1
                            continue

                    jobs.append(_normalize_job(job, posted_at_dt))
                    if len(jobs) >= limit:
                        break

                # /search-jobs documents `start` in 25-result increments; a short page means we're done.
                if len(batch) < SEARCH_JOBS_PAGE_SIZE:
                    break

                # When sorted by recency, stop once an entire parseable page is older than the cutoff.
                if cutoff is not None and parseable_dates and max(parseable_dates) < cutoff:
                    break

                start += SEARCH_JOBS_PAGE_SIZE

            result = {
                "query": {
                    "keywords": keywords,
                    "location": location,
                    "location_id": location_id,
                    "max_age_days": max_age_days,
                    "onsite_remote": onsite_remote,
                },
                "count": len(jobs),
                "jobs": jobs,
                "meta": {
                    "requested_limit": limit,
                    "pages_fetched": pages_fetched,
                    "raw_jobs_seen": raw_seen,
                    "duplicates_removed": duplicates_removed,
                    "age_filtered_out": age_filtered_out,
                },
            }

            return result

    except Exception as e:
        logger.exception("Error searching jobs")
        error_msg = f"Error searching jobs: {e}"
        return {"error": error_msg} if not format_output else error_msg


@mcp.tool()
def get_job_details(job_id: str) -> dict:
    """
    Get detailed information about a specific LinkedIn job posting.

    :param job_id: The LinkedIn job ID
    :return: Detailed job information
    """
    try:
        with get_client() as client:
            response = client.get(f"{API_BASE_URL}/get-job-details", params={"id": job_id})
            data = _parse_api_response(response)
            return data.get("data", {})
    except Exception as e:
        logger.exception("Error fetching job details")
        return {"error": f"Error fetching job details: {e}"}


@mcp.tool()
def search_locations(keyword: str) -> str:
    """
    Search for LinkedIn location IDs by keyword.

    :param keyword: Location keyword to search for
    :return: ID of the first matching location
    """
    try:
        with get_client() as client:
            location_id = _lookup_location_id(client, keyword)
            print(f"Location ID: {location_id}")
            return location_id
    except Exception as e:
        logger.exception("Error searching locations")
        return f"Error searching locations: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
