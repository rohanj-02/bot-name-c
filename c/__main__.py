import os
import aiohttp
from aiohttp import web
from gidgethub import routing, sansio
from gidgethub import aiohttp as gh_aiohttp

router = routing.Router()


@router.register("issues", action="opened")
async def issue_opened_event(event, gh, *args, **kwargs):
	"""
	Event when an issue is opened
	"""
	## TODO
	url = event.data["issue"]["comments_url"]
	# Get the comment url from the event data
	# After getting the url, set it to the variable url
	# and push the changes.
    name = event.data["issue"]["user"]["login"]
	message = "Hello " + name
	await gh.post(url, data={"body": message})

async def main(request):
	body = await request.read()

	secret = os.environ.get("GH_SECRET")
	oauth_token = os.environ.get("GH_AUTH")

	event = sansio.Event.from_http(request.headers, body, secret=secret)
	async with aiohttp.ClientSession() as session:
		gh = gh_aiohttp.GitHubAPI(session, "rohanj-02",
								  oauth_token=oauth_token)
		# Enter your username above in <USERNAME> field
		await router.dispatch(event, gh)
	return web.Response(status=200)


if __name__ == "__main__":
	app = web.Application()
	app.router.add_post("/", main)
	port = os.environ.get("PORT")
	if port is not None:
		port = int(port)

	web.run_app(app, port=port)
