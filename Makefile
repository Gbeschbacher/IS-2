CONFIG = config.p
RESULT = results.png

USERS = "./data/users/"
OVERALL = "./data/overall/"
BASE_USERS = "./data/base.csv"

LISTENING_EVENTS = "./scripts/listening-events.py"
USER_FETCHER = "./scripts/user-fetcher.py"
CONVERTER_UAM = "./scripts/converter-uam.py"
MINIMUM_USERS = 10

UNIQUE_USERS = ./data/overall/unique_users_2000.csv
LISTENING_EVENTS_F = ./data/overall/listening_events_34918.csv

all: paths users listening-events

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


clean.users:
	rm -rf ./$(USERS)

clean.overall:
	rm -rf ./$(OVERALL)

clean: clean.users clean.overall
