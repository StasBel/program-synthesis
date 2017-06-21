from github import Github

if __name__ == "__main__":
    g = Github("StasBel", "cog1to3rgo5um")
    repos = g.search_repositories(query="language=C, fork=false, stars>=0")
    for repo in repos:
        print(repo.type)
