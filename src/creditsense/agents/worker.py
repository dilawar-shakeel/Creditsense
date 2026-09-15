from creditsense.config import get_settings


def main() -> None:
    settings = get_settings()
    print(f"{settings.agent_name} is ready.")


if __name__ == "__main__":
    main()
