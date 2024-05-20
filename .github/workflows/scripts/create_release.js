// Uses Github's API to create the release and wait for result.
// We use a JS script since github CLI doesn't provide a way to wait for the release's creation and returns immediately.

module.exports = async (github, context, core) => {
	try {
		const response = await github.rest.repos.createRelease({
			draft: false,
			generate_release_notes: true,
			name: process.env.RELEASE_TAG,
			owner: context.repo.owner,
			prerelease: true,
			repo: context.repo.repo,
			tag_name: process.env.RELEASE_TAG,
		});

		core.setOutput('upload_url', response.data.upload_url);
	} catch (error) {
		core.setFailed(error.message);
	}
}