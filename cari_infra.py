import asyncio
import re
from scrapling.fetchers import AsyncStealthySession


async def main():
    async with AsyncStealthySession(
        headless=True, solve_cloudflare=True, max_pages=1,
    ) as session:
        page = await session.fetch("https://www.idnfinancials.com/tlkm/pt-telkom-indonesia-persero-tbk")

        if not page.body:
            print("HTML kosong")
            return

        html = page.body.decode("utf-8", errors="ignore")
        print(f"Panjang HTML: {len(html)}")
        print()

        # Cari kata "Infrastructure" (dengan s atau tanpa s)
        for keyword in ["Infrastructure", "Infrastructures",
                        "Telecommunication", "Telecommunications"]:
            print("=" * 70)
            print(f"Cari: {keyword}")
            print("=" * 70)

            positions = [m.start() for m in re.finditer(keyword, html, re.IGNORECASE)]

            if not positions:
                print("  Tidak ditemukan")
            else:
                print(f"  Ditemukan: {len(positions)}x")
                for i, pos in enumerate(positions[:3], 1):
                    start = max(0, pos - 200)
                    end = min(len(html), pos + 200)
                    snippet = html[start:end]
                    print(f"\n  --- Kejadian {i} (pos {pos}) ---")
                    print(f"  {snippet}")
            print()


if __name__ == "__main__":
    asyncio.run(main())