CONFIG = config.p
RESULT = results.png

USERS = "./data/users/"
ARTISTS = "./data/artists/"
OVERALL = "./data/overall/"
BASE_USERS = "./data/base.csv"

LISTENING_EVENTS = "./scripts/listening-events.py"
USER_FETCHER = "./scripts/user-fetcher.py"
ARTIST_INFO_FETCHER = "./scripts/artist-info-fetcher.py"
CONVERTER_UAM = "./scripts/converter-uam.py"
WEB_SIMILARITY = "./scripts/web-similarity.py"
MINIMUM_USERS = 10

UNIQUE_USERS = "./data/overall/unique_users_2000.csv"
LISTENING_EVENTS_F = "./data/overall/listening_events_901767.csv"
UAM_ARTISTS = "./data/overall/UAM_artists.csv"

plot:
	gnuplot $(CONFIG) > $(RESULT)

paths:
	if test -d $(USERS); then echo "Directory $(USERS) already exists"; else mkdir $(USERS); fi
	if test -d $(OVERALL); then echo "Directory $(OVERALL) already exists"; else mkdir $(OVERALL); fi

users:
	python $(USER_FETCHER) $(BASE_USERS) $(OVERALL) $(MINIMUM_USERS)

listening-events:
	python $(LISTENING_EVENTS) $(UNIQUE_USERS) $(OVERALL)

uam:
	python $(CONVERTER_UAM) $(LISTENING_EVENTS_F) $(OVERALL)

artist-info:
	python $(ARTIST_INFO_FETCHER) $(ARTISTS) $(UAM_ARTISTS)

web-similarity:
	python $(WEB_SIMILARITY) $(ARTISTS) $(UAM_ARTISTS) $(OVERALL)

clean.users:
	rm -rf ./$(USERS)

clean.overall:
	rm -rf ./$(OVERALL)

clean: clean.users
