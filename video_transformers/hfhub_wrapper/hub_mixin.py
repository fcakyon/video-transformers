import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from huggingface_hub import hf_api
from huggingface_hub.hf_api import HfApi, HfFolder
from huggingface_hub.repository import Repository

from video_transformers.templates import export_hf_model_card


def push_to_hub(
    self,
    # NOTE: deprecated signature that will change in 0.12
    *,
    repo_path_or_name: Optional[str] = None,
    repo_url: Optional[str] = None,
    commit_message: Optional[str] = "Add model",
    organization: Optional[str] = None,
    private: bool = False,
    api_endpoint: Optional[str] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
    git_user: Optional[str] = None,
    git_email: Optional[str] = None,
    config: Optional[dict] = None,
    skip_lfs_files: bool = False,
    # NOTE: New arguments since 0.9
    repo_id: Optional[str] = None,  # optional only until 0.12
    token: Optional[str] = None,
    branch: Optional[str] = None,
    create_pr: Optional[bool] = None,
    allow_patterns: Optional[Union[List[str], str]] = None,
    ignore_patterns: Optional[Union[List[str], str]] = None,
    # TODO (release 0.12): signature must be the following
    # repo_id: str,
    # *,
    # commit_message: Optional[str] = "Add model",
    # private: bool = False,
    # api_endpoint: Optional[str] = None,
    # token: Optional[str] = None,
    # branch: Optional[str] = None,
    # create_pr: Optional[bool] = None,
    # config: Optional[dict] = None,
    # allow_patterns: Optional[Union[List[str], str]] = None,
    # ignore_patterns: Optional[Union[List[str], str]] = None,
) -> str:
    """
    Upload model checkpoint to the Hub.

    Use `allow_patterns` and `ignore_patterns` to precisely filter which files
    should be pushed to the hub. See [`upload_folder`] reference for more details.

    Parameters:
        repo_id (`str`, *optional*):
            Repository name to which push.
        commit_message (`str`, *optional*):
            Message to commit while pushing.
        private (`bool`, *optional*, defaults to `False`):
            Whether the repository created should be private.
        api_endpoint (`str`, *optional*):
            The API endpoint to use when pushing the model to the hub.
        token (`str`, *optional*):
            The token to use as HTTP bearer authorization for remote files.
            If not set, will use the token set when logging in with
            `transformers-cli login` (stored in `~/.huggingface`).
        branch (`str`, *optional*):
            The git branch on which to push the model. This defaults to
            the default branch as specified in your repository, which
            defaults to `"main"`.
        create_pr (`boolean`, *optional*):
            Whether or not to create a Pull Request from `branch` with that commit.
            Defaults to `False`.
        config (`dict`, *optional*):
            Configuration object to be saved alongside the model weights.
        allow_patterns (`List[str]` or `str`, *optional*):
            If provided, only files matching at least one pattern are pushed.
        ignore_patterns (`List[str]` or `str`, *optional*):
            If provided, files matching any of the patterns are not pushed.

    Returns:
        The url of the commit of your model in the given repository.
    """
    # If the repo id is set, it means we use the new version using HTTP endpoint
    # (introduced in v0.9).
    if repo_id is not None:
        token, _ = hf_api._validate_or_retrieve_token(token)
        api = HfApi(endpoint=api_endpoint)

        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            token=token,
            private=private,
            exist_ok=True,
        )

        # Push the files to the repo in a single commit
        with tempfile.TemporaryDirectory() as tmp:
            saved_path = Path(tmp) / repo_id
            self.save_pretrained(saved_path, config=config)
            export_hf_model_card(
                export_dir=saved_path,
                labels=self.labels,
                backbone_config=self.config["backbone"],
                neck_config=self.config["neck"],
                preprocessor_config=self.config["preprocessor"],
                head_config=self.config["head"],
                total_model_params=self.num_total_params,
                total_trainable_model_params=self.num_trainable_params,
            )
            return api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                token=token,
                folder_path=saved_path,
                commit_message=commit_message,
                revision=branch,
                create_pr=create_pr,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )

    # If the repo id is None, it means we use the deprecated version using Git
    # TODO: remove code between here and `return repo.git_push()` in release 0.12
    if repo_path_or_name is None and repo_url is None:
        raise ValueError("You need to specify a `repo_path_or_name` or a `repo_url`.")

    if use_auth_token is None and repo_url is None:
        token = HfFolder.get_token()
        if token is None:
            raise ValueError(
                "You must login to the Hugging Face hub on this computer by typing"
                " `huggingface-cli login` and entering your credentials to use"
                " `use_auth_token=True`. Alternatively, you can pass your own token"
                " as the `use_auth_token` argument."
            )
    elif isinstance(use_auth_token, str):
        token = use_auth_token
    else:
        token = None

    if repo_path_or_name is None:
        repo_path_or_name = repo_url.split("/")[-1]

    # If no URL is passed and there's no path to a directory containing files, create a repo
    if repo_url is None and not os.path.exists(repo_path_or_name):
        repo_id = Path(repo_path_or_name).name
        if organization:
            repo_id = f"{organization}/{repo_id}"
        repo_url = HfApi(endpoint=api_endpoint).create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            repo_type=None,
            exist_ok=True,
        )

    repo = Repository(
        repo_path_or_name,
        clone_from=repo_url,
        use_auth_token=use_auth_token,
        git_user=git_user,
        git_email=git_email,
        skip_lfs_files=skip_lfs_files,
    )
    repo.git_pull(rebase=True)

    # Save the files in the cloned repo
    self.save_pretrained(repo_path_or_name, config=config)
    export_hf_model_card(
        export_dir=saved_path,
        labels=self.labels,
        backbone_config=self.config["backbone"],
        neck_config=self.config["neck"],
        preprocessor_config=self.config["preprocessor"],
        head_config=self.config["head"],
        total_model_params=self.num_total_params,
        total_trainable_model_params=self.num_trainable_params,
    )

    # Commit and push!
    repo.git_add(auto_lfs_track=True)
    repo.git_commit(commit_message)
    return repo.git_push()
