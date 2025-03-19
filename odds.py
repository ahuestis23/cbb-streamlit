import requests
import json
import time
import datetime
import pandas as pd
import asyncio
import aiohttp

cookies = {
    'intercom-id-xgkfzdmt': '2d150861-e37b-49b3-9fd8-93b235e974ff',
    'intercom-device-id-xgkfzdmt': 'beeab95d-49dc-45e2-90db-2fd10bc586b5',
    'uwguid': 'WEBLS-e3266bb5-37f2-4fe8-95e2-b82b098c9fb5',
    '__stripe_mid': '1497676f-8824-4e19-aa02-d967ba18c6ff91f063',
    'analytics_session_id': '1726158240405',
    'analytics_session_id.last_access': '1726158242048',
    'AMP_MKTG_a38e6c5e34': 'JTdCJTdE',
    'AMP_a38e6c5e34': 'JTdCJTIyZGV2aWNlSWQlMjIlM0ElMjIzN2M3MzUwZS1hMTFkLTQyMzItYTFmNi0yZGYyNzNmNTIyZDYlMjIlMkMlMjJzZXNzaW9uSWQlMjIlM0ExNzI2Mzg2MjY4Nzc4JTJDJTIyb3B0T3V0JTIyJTNBZmFsc2UlMkMlMjJsYXN0RXZlbnRUaW1lJTIyJTNBMTcyNjM4NjI2ODgxNiUyQyUyMmxhc3RFdmVudElkJTIyJTNBMTA3MCUyQyUyMnBhZ2VDb3VudGVyJTIyJTNBMSU3RA==',
    'ph_phc_jhnVVaMuz0pOcWWu39hTCdQJpy1DmiZXbzwK6KwwcXf_posthog': '%7B%22distinct_id%22%3A%220191d817-d488-7fe6-a862-6b541f5093fb%22%2C%22%24sesid%22%3A%5B1726386272442%2C%220191f4a3-e9aa-734d-8101-cf9693759a21%22%2C1726386268586%5D%7D',
    'access_token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzYzODQxNDk3LCJqdGkiOiJmYWUzZjQzNGU2MTM0YWMyOTk5YmM0MjdjYmMyYWM4YyIsInVzZXJfaWQiOjkxNTcxLCJwbGFucyI6WyJwbGF0aW51bSJdLCJiYW5rcm9sbCI6MjUwMC4wfQ.tnnujKiVTil6CCNOglPX4dIpRZOD5it9AEQWqzumEPw',
    'cookieyes-consent': 'consentid:dklFU3doWjFRUzl4aWVPbjlRZE5lRElLdloxdW5FWHQ,consent:yes,action:no,necessary:yes,functional:no,analytics:no,performance:no,advertisement:no,other:no',
    #'local/ab-test/silver-plan-experiment': 'Control',
    #'local/ab-test/gold249_inseason': 'Control',
    #'local/ab-test/homepage-experiment': 'Treatment',
    #'local/ab-test/logged-in-homepage-experiment': 'Control',
    'cf_clearance': 'Kq3cKWKQl9946j1AHk4L0xa4TM0uOSKyvJF_LfwYH00-1741916404-1.2.1.1-vmEmpb.N.4L3FTfibMPCRZ23Wy_C.kGVlIqLQo3SkZSiecLaM.qqllhl7a287tWOMouwmarQPyh1YbUxAunVI7Tvicmx4mAAS_R1Zc_R_937pMj1xDCHaE0AbzWqdb8BG.WW2M.JDKGWIflaFhCFM7eIvb.7WQ5e_MRbLAUjZTpQ1fM38ljjyvCZRAqXScnJFM1jf.PtSATppIlqdRWRz8p91jr.8ktDh0zaJL8gIi6jg6dRKYHAtl9xGTTAKo6THgN3QHQFjju_Jp9Olp7qAU3n6RojDits2ISyay5pvIkoasiL9u_OrDWL3hPip4Sa.ObLOC35mxNbfsw11Kiwd.SErX5CXJBM.kp.T8cwYiI',
    #'local/ab-test/OJ-15-NewCheckout-allDevices': 'Control',
    #'local/ab-test/OJ-17-globalplanpricing-allDevices': 'Control',
    #'local/ab-test/OJ-17-BettingToolVideos-allDevices': 'Control',
    #'local/ab-test/OJ-14-pandppricing-allDevices': 'Treatment249',
    'state': '',
    'onboarding-v2-widget-expanded': 'false',
    'sidebar': 'false',
    'navGroups': '{}',
    'HeardAboutUs': '',
    'heardAboutUsSource': '',
    'heardAboutUsSpecific': '',
    '__stripe_sid': '9abb63e6-42a6-49ec-8045-5be0fc878a7bae2c8f',
}

headers = {
    'accept': 'application/json',
    'accept-language': 'en-US,en;q=0.9',
    'baggage': 'sentry-environment=preview,sentry-release=jjVBBro_gvzbMQvNsfC6E,sentry-public_key=66796a2fc7244437a5d67cc60232438b,sentry-trace_id=9de58385135446cfb63fed6057675309,sentry-sample_rate=0,sentry-transaction=%2F%5BsportOrLeague%5D%2Fscreen%2F%5Bmarket%5D,sentry-sampled=false',
    'content-type': 'application/json',
    'priority': 'u=1, i',
    'referer': 'https://oddsjam.com/ncaab/screen/player_assists',
    'sec-ch-ua': '"Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'sentry-trace': '9de58385135446cfb63fed6057675309-ba7beaa7dbde70c8-0',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36',
}

last_reset_time = datetime.datetime.now()
processed_ids = set()

def get_params(market):
  params = {
      'sport': 'basketball',
      'league': 'ncaab',
      'state': 'AZ',
      'market_name': market,
      'is_future': '0',
      'game_status_filter': 'Pre-Match',
  }
  return params

class Discrepancies:
    def __init__(self):
        self.data_list = []

    async def get_discrepancies(self):
        self.data_list = []
        markets = ['player_points','player_assists', 'player_rebounds']
        async with aiohttp.ClientSession() as session:
            tasks = [self.get_market_data(session, market) for market in markets]
            await asyncio.gather(*tasks)
        return pd.DataFrame(self.data_list)

    async def fetch_data(self, session, market):
        async with session.get(
            'https://oddsjam.com/api/backend/oddscreen/v2/game/data',
            params=get_params(market),
            cookies=cookies,
            headers=headers
        ) as response:
            return await response.json(content_type=None)

    async def get_market_data(self, session, market):
        api_response = await self.fetch_data(session, market)
        for item in api_response["data"]:
            game_id = item["game_id"]
            id = item["id"]
            team_name = item["playerTeamName"]
            oj_markets = {
                'player_points': 'Player Points',
                'player_assists': 'Player Assists',
                'player_rebounds': 'Player Rebounds'
            }
            clean_market_name = oj_markets[market]
            sportsbooks_to_include = ["FanDuel", "BetOnline", "Fliff"]
            for row in item["rows"]:
                title = row["display"][clean_market_name]["title"]
                player_name = row["display"][clean_market_name]["player_name"]
                for sportsbook, odds_data in row["odds"].items():
                    if sportsbook in sportsbooks_to_include:
                        if not odds_data:
                            continue
                        for odds_entry in odds_data:
                            date = game_id.split("-")[2] + "-" + game_id.split("-")[3] + "-" + game_id.split("-")[4]
                            self.data_list.append({
                                "id": id,
                                "game_id": game_id,
                                "title": title,
                                "player_name": player_name,
                                "team_name": team_name,
                                "sportsbook": sportsbook,
                                "price": odds_entry.get("price", "NA"),
                                "market_name": odds_entry["market_name"],
                                "bet_name": odds_entry["name"],
                                "bet_points": odds_entry["bet_points"],
                                "player_id": odds_entry["player_id"],
                                "date": date
                            })

async def main():
    Discreps = Discrepancies()
    df = await Discreps.get_discrepancies()
    df['new_id'] = df['id'] + '_' + df['market_name']+ '_' + df['sportsbook']
    grouped = df.pivot(index=['new_id', 'sportsbook', 'market_name'], columns='title', values='price')
    grouped.reset_index(inplace=True)
    grouped.columns.name = None  # Remove the index name
    grouped.columns = ['new_id', 'sportsbook', 'market_name'] + grouped.columns.tolist()[3:]
    merged = grouped.merge(df[['new_id', 'sportsbook', 'market_name', 'bet_name', 'bet_points', 'player_id', 'player_name', 'team_name', 'date']], on=['new_id', 'sportsbook', 'market_name'])
    merged = merged.drop_duplicates(subset=['new_id', 'sportsbook', 'bet_points', 'market_name'])
    cols = ['date', 'player_name', 'team_name', 'market_name', 'sportsbook', 'bet_points', 'Over', 'Under']
    merged = merged[cols]
    merged.to_csv('odds.csv', index=False)

if __name__ == "__main__":
    asyncio.run(main())
