# AAM:

    - durch data-mining im web für jeden artist die gefundenen terms angesehen und versucht auf 2 verschiedene arten zu reduzieren:
    1.) minimum dcument frequency
        - seltene terms (die, die nur bei wenigen artists vorkommen)  wollten wir weglassen; kann aber dazu führen, dass gerade diese terms eine große ausiwrkung bei der berechnung der ähnlichkeiten der artists untereinander haben.
        - terms, die nur bei einem artist vorkommen, sind vernachlässigbar, da diese die distanz zu allen anderen artists nur konstant erhöhen und keinen weiteren mehrwert bieten;

        ==> durch das entfernen von termen die nur bei einem artist vorkomen, konnten wir die termzahl von X auf Y reduzieren

    2.) stopwords expansion
    dadurch, dass wir immer von gleichen quellen fetchen (wikipedia, discogs, audiodb), gibt es bestimmt terme, die bei sehr vielen artists vorkommen, aber nicht wirklich für artists relevant sind (bsp: Wikipedia, page, link u dgl.); wir hätten de terme herausfinden können und zur liste an stopwords hinzufügen können um sie somit von der analyse auszuschließen. dabei hätten wir das totale vorkommen jedes terms in den gesamten daten aufsummieren müssen und uns die häufigsten ansehen müssen. wir haben uns allerdings dagegen entschieden, zum einen weil die laufzeit ohnehin schon sehr hoch war und andererseits, weil die entscheidung, ob dieses stopword wirklich relevant ist für die unterscheidung der artists oder nicht, sehr subjektiv ist; mithilfe einer PCA könnte man es auf jene terme reduzieren, die die artists am besten unterscheiden

    3.) term index lookup:
    um tfidf berechnung schneller zu machen, haben wir einen lookup table für terms erstellt, um so möglichst effizient den index zu einem bestimmten term zu finden.

    for t in terms:
            if (termLookup.has_key(t)):
                tIdx = termLookup[t]
                tfidf[aIdx, tIdx] += 1

    4.) CB Recommender:
        CB aus der vorlesung erweitert (den aus der vorlesung vl. nochmal testen und dann vergleichen?^^ vl. hab ich hier blödsinn gmacht, glaub aber ned);

        anstatt der average similarity beim scoring der neirest neighbors haben wir die summe der similarities für jeden unique artistIDX aus den neirest neighbors verwendet

        for nidx in uniq_neighbor_idx:
            mask = np.where(neighbor_idx == nidx)
            sims_list_for_curr_nidx = sims_neighbors_idx[mask]

            sum_sim = np.sum(sims_list_for_curr_nidx)
            # if(len(sims_list_for_curr_nidx) > max_length):
            #     max_length = len(sims_list_for_curr_nidx)
            dict_recommended_artists_idx[nidx] = sum_sim

        weil summe > 1 sein kann, is sie noch mit den similarity-werten normalisiert worden um somti zwischen 0 und 1 zu sein;
        auch beschränken wir die anzahl an recommended artists; wir achten jedoch nicht draauf, dass diese anzahl immer erreicht wird

        sorted_recommended_artists_key_value = sorted(dict_recommended_artists_idx.items(), key=operator.itemgetter(1), reverse = True)
        max_value = sorted_recommended_artists_key_value[0][1]
        recommended_artists_idx_all = {}
        count = 1
        for (key, value) in sorted_recommended_artists_key_value:
            recommended_artists_idx_all[key] = float(float(value) / float(max_value))
            if count >= max_artists:
                break
            count += 1

    5.)  CF REcommender:
        erweitert, damit man ihn mit CB vergleichen kann: scoring wurde aus der häufigkeit jedes artists in den k-nearest-neighbors ermittelt; weiters wurde performance des scoring algorithmus durch gewichungsfunktion verbessert, die rücksicht auf die reihenfolge/distanz der nearest neighbors zm aktuellen user berücksichtigt


        recommended_artists_idx_all_dic = {}
        x = 1
        y = 1
        for neighbor_idx in neighbor_indices:
            # Compute the set difference between seed user's neighbor and seed user,
            # i.e., artists listened to by the neighbor, but not by seed user.
            # These artists are recommended to seed user.

            artist_idx_n = np.nonzero(UAM[neighbor_idx, :])     # indices of artists user u's neighbor listened to
            # np.nonzero returns a tuple of arrays, so we need to take the first element only when computing the set difference
            recommended_artists_idx = np.setdiff1d(artist_idx_n[0], artist_idx_u)

            for idx in recommended_artists_idx:
                if(not recommended_artists_idx_all_dic.has_key(idx)):
                    recommended_artists_idx_all_dic[idx] = y
                else:
                    recommended_artists_idx_all_dic[idx] += y

            y = (x * x) / 2 + 1
            x += 1

        --> nachher wieder normalisiert weil ie werte wieder > 1 sein können


    6.) bonus: popular items recommender:
        step 1 - mehr daten:
            um dies zu implementieren, braucht man viel mehr daten als bisher (lokalisierungsdaten der user, länderspezifiscieh charts, etc.). theoretisch wäre es über die last.fm api möglich, von user die countries zu bekommen und von jedem country die metropolen (geo.getmetros). mit diesen infos kann man dann geo.getmetroweeklychartlist  crawlen und die infos mergen.
        step 2 - mergen:
            die artists aus den charts müssen in bezug zu den artists gesett werden, die der user bereits gehört hat. hier würde wahrscheinlich eine artist-charts:artists-users similarity-matrix sinnvoll sein.

- Probleme:
    Google crawlen über python ist nicht so einfach, da keine suchergebnisse angefordert werden können



### PART 3:

1. User-based Similarity Measure

Ausgangsbasis: C1ku_users_extended.csv
zur similarity berechnung haben wir folgende maße verwendet:
- geodäsische längendistanz: user im selben land oder in einem nahen land sind sich ähnlicher; (wurde mit geopy und den gegebenen koordinaten des herkunftslandes berechnet)
- age: user mit geringem altersunterschied sind sich ähnlicher
- gender: user mit gleichem geschlecht sind sich ähnlicher
keine daten: user, die keine bestimmten daten angegeben haben sind sich ähnlicher

alle werte wurden normalisiert und mit einer gleichverteilten gewichtung von 0.33 aufsummiert, um den sim.-wert zwischen zwei usern zu ermitteln

2. user based recommender
selbstständiger user-based recommender wurde ähnlich zum bereits vorhandenen cb-recommender implementiert (einzige unterschied ist UUM statt AAM)

3. extended col. filtering
CF approach wurde mit UUM erweitert;

3.1 user similarity aus der UAM ausmultipliziert mit user similarity aus UUM (Usersimilarity UAM * UUM: sim_users[u] = sim * UUM[seed_uidx, u]), wobei sim der ausgerechneten similarity (aus der UAM) entspricht.
User similarity UUM-Threshhold

3.2 ein user similarity cutoff mittels threshhold
    if UUM[seed_uidx, u] < 0.7:
        sim_users[u] = 0

Für alle user, die in der UUM einen niedrigeren sim.-wert haben, wird die gesamt similarity auf 0 gesetzt, um nicht mehr als empfehlungsvorschlag in frage zu kommen

conclusio von 3.1 und 3.2 vom testen her funktioniert keiner der beiden besser und keiner der ansätze kann weder als verbesserung, noch als verschlechterung aufkommen

4, hybrid user-based und cf

wie vorher (scoring based ranking), wo CF mit 100% gewichtet wurde und UB mit 20%;

normalisierung dre werte wie vorheris nicht notwendig, da dies beim sortieren bedeutungslos ist

5. UB recommender, extended CF, HR_UBCF_SB

knn mindestens 20, aber k kann dynamisch wachsen, falls die geforderte anzahl an zu recommendeten artists nicht erreicht wird
