# Tests complets pour le RAG local

Ce fichier contient une liste de questions pour tester séparément les trois documents :

1. `rag_test_exemple.txt` — Bibliothèque universitaire / réseau / sécurité
2. `rag_test_nouveau_contexte.md` — Boutique NovaTech
3. `rag_test_centre_formation.pdf` — Centre Horizon

---

# 1. Tests — rag_test_exemple.txt

## A. Questions de récupération directe

1. Combien d'ordinateurs possède la bibliothèque ?
2. Combien d'ordinateurs sont dans la salle A ?
3. Combien d'ordinateurs sont dans la salle B ?
4. Combien d'ordinateurs sont dans la salle C ?
5. Combien de personnes travaillent au service informatique ?
6. Combien y a-t-il d'administrateurs systèmes ?
7. Combien y a-t-il de techniciens réseau ?
8. Combien y a-t-il de développeurs ?
9. Combien y a-t-il de responsables informatiques ?
10. À quelle heure les sauvegardes quotidiennes sont-elles réalisées ?
11. Quel jour est réalisée la sauvegarde complète ?
12. Pendant combien de jours les sauvegardes sont-elles conservées ?
13. Pendant combien de jours les journaux de sécurité sont-ils conservés ?
14. Quelle est l'adresse du réseau administratif ?
15. Quelle est l'adresse du réseau étudiant ?
16. Quelle est l'adresse du réseau des serveurs ?
17. Quelle est l'adresse IP du serveur web ?
18. Quelle est l'adresse IP du serveur de fichiers ?

## B. Questions de compréhension

19. Quelles sont les cinq étapes générales du fonctionnement d'un RAG ?
20. Pourquoi un système RAG utilise-t-il des chunks ?
21. À quoi servent les embeddings dans un RAG ?
22. Quel est le rôle de la recherche de passages pertinents ?
23. Pourquoi les étudiants ne doivent-ils pas avoir accès directement au réseau administratif ?
24. Quel est le rôle du pare-feu dans cette architecture ?
25. Quelle règle concerne les mots de passe ?
26. Qui doit obligatoirement utiliser l'authentification multifacteur ?
27. À quelle fréquence les comptes utilisateurs sont-ils révisés ?
28. Que s'est-il passé le 15 juin ?
29. D'où provenaient les tentatives de connexion inhabituelles ?
30. Quelle mesure a été prise au niveau du pare-feu ?
31. Qu'est-il arrivé au compte concerné ?
32. Quelle différence existe entre une sauvegarde quotidienne et une sauvegarde complète ?

## C. Questions nécessitant plusieurs informations

33. Combien d'ordinateurs sont présents dans les salles A et B réunies ?
34. Combien d'ordinateurs sont présents dans les salles B et C réunies ?
35. Combien d'ordinateurs sont présents dans les salles A et C réunies ?
36. Les trois salles représentent-elles bien les 500 ordinateurs de la bibliothèque ?
37. Quelle est la composition complète du service informatique ?
38. Quels sont les trois segments du réseau et à quoi correspondent-ils ?
39. Quelles sont les principales mesures de sécurité appliquées aux utilisateurs et aux réseaux ?
40. Quelles sont les différentes durées de conservation mentionnées dans le document ?

## D. Questions auxquelles le RAG doit pouvoir répondre sans inventer

41. Quelle est l'adresse IP du serveur DNS ?
42. Quelle est l'adresse IP du serveur de messagerie ?
43. Combien d'ordinateurs possède la salle D ?
44. Quel est le nom du pare-feu utilisé ?
45. Quel système d'exploitation utilisent les serveurs ?
46. Quel est le nom du responsable informatique ?
47. Quel est le budget annuel de la bibliothèque ?
48. Quelle est la date exacte de la prochaine sauvegarde ?

Pour ces questions, l'information n'est pas fournie dans le document. Le RAG devrait indiquer que l'information est absente plutôt que l'inventer.

---

# 2. Tests — rag_test_nouveau_contexte.md

## A. Questions de récupération directe

1. Quel est le nom de l'entreprise ?
2. Quelle est l'activité de NovaTech ?
3. Quels sont les horaires d'ouverture de la boutique ?
4. Quels sont les trois départements de l'entreprise ?
5. Combien de personnes travaillent dans le département Vente ?
6. Combien de personnes travaillent dans le département Stock ?
7. Combien de personnes travaillent dans l'Administration ?
8. Combien de personnes travaillent au total chez NovaTech ?
9. Qui supervise les trois départements ?
10. Quel jour les inventaires sont-ils réalisés ?
11. À quel moment de la journée les inventaires sont-ils réalisés ?
12. Quel est le seuil minimal pour les produits courants ?
13. Quel est le seuil minimal pour les produits coûteux ?
14. Que fait le responsable du stock lorsqu'un produit atteint son seuil minimal ?
15. Quel jour les commandes fournisseurs sont-elles généralement préparées ?
16. Quel est le délai moyen de livraison des fournisseurs ?
17. Qui vérifie les factures fournisseurs ?
18. Quels sont les produits les plus vendus ?
19. Quel est le prix moyen d'un clavier ?
20. Quel est le prix moyen d'une souris ?
21. Quel est le prix moyen d'un casque audio ?
22. Combien de jours un client peut-il retourner un produit ?
23. Quelle condition doit être respectée pour effectuer un retour ?
24. Que peut-il arriver à un produit présentant un défaut de fabrication ?

## B. Questions de compréhension

25. Pourquoi réalise-t-on régulièrement un inventaire ?
26. Que signifie le seuil minimal de réapprovisionnement ?
27. Quelle différence existe entre le seuil des produits courants et celui des produits coûteux ?
28. Que se passe-t-il lorsqu'un produit atteint son seuil minimal ?
29. Pourquoi les factures fournisseurs sont-elles vérifiées ?
30. Quelle est la procédure prévue pour un produit présentant un défaut de fabrication ?
31. Quels produits semblent être les plus importants pour les ventes de NovaTech ?
32. Quelles sont les conditions de retour d'un produit ?

## C. Questions nécessitant plusieurs informations

33. Combien de personnes travaillent dans les départements Vente et Stock réunis ?
34. Combien de personnes travaillent dans les départements Stock et Administration réunis ?
35. Combien de personnes ne travaillent pas dans le département Vente ?
36. Quel est le prix moyen total d'un clavier et d'une souris ?
37. Quel est le prix moyen total d'une souris et d'un casque audio ?
38. Quel est le prix moyen total d'un clavier et d'un casque audio ?
39. Si un client achète un clavier et une souris, quel est le prix moyen total ?
40. Si un produit courant possède 10 unités en stock, que doit faire le responsable du stock ?
41. Si un produit coûteux possède 5 unités en stock, que doit faire le responsable du stock ?
42. Résume les principales règles concernant le stock, les commandes et les retours clients.

## D. Questions auxquelles le RAG ne doit pas inventer

43. Quel est le chiffre d'affaires annuel de NovaTech ?
44. Combien coûte un ordinateur portable ?
45. Quel est le nom du responsable de la boutique ?
46. Combien de fournisseurs possède NovaTech ?
47. Quelle est l'adresse de NovaTech ?
48. Quel est le numéro de téléphone de la boutique ?
49. Combien de produits différents sont disponibles en stock ?
50. Quel est le salaire moyen des employés ?
51. Quelle est la marque des claviers vendus ?
52. Quelle est la date de création de NovaTech ?

Ces informations ne sont pas présentes dans le document.

---

# 3. Tests — rag_test_centre_formation.pdf

## A. Questions de récupération directe

1. Quel est le nom du centre de formation ?
2. Quel type d'établissement est le Centre Horizon ?
3. Quelles formations sont proposées ?
4. Combien de salles possède le centre ?
5. Combien de postes informatiques possède la salle Alpha ?
6. Combien de postes informatiques possède la salle Beta ?
7. Combien de postes informatiques possède la salle Gamma ?
8. Combien de postes informatiques possède la salle Delta ?
9. À quelle heure commencent les cours du matin ?
10. À quelle heure se terminent les cours du matin ?
11. À quelle heure commencent les cours de l'après-midi ?
12. À quelle heure se terminent les cours de l'après-midi ?
13. Combien de temps dure la pause pendant chaque session ?
14. Combien de semaines dure la formation Développement Web ?
15. Combien de semaines dure la formation Bases de Données ?
16. Combien de semaines dure la formation Administration Réseau ?
17. Quels types d'évaluations sont prévus ?
18. Quel pourcentage de la note globale représente le projet final ?
19. Quel pourcentage représente les travaux pratiques ?
20. Quand les étudiants peuvent-ils utiliser les ordinateurs du centre ?
21. Que doivent faire les étudiants après chaque séance ?
22. À qui faut-il signaler un problème matériel ?
23. Quel document doit être remis avant le début de la formation ?
24. Comment les candidats reçoivent-ils la confirmation de leur inscription ?

## B. Questions de compréhension

25. Quels sont les domaines de formation proposés par le centre ?
26. Quelle différence existe entre les formations en termes de durée ?
27. Pourquoi les étudiants doivent-ils signaler les problèmes matériels ?
28. Comment est calculée la note globale ?
29. Quelle est l'importance du projet final dans l'évaluation ?
30. Quelles sont les principales règles concernant l'utilisation des salles ?
31. Quelle est la procédure générale d'inscription ?
32. Quel est le rôle de la preuve de confirmation envoyée par courrier électronique ?

## C. Questions nécessitant plusieurs informations

33. Combien de postes informatiques y a-t-il dans les quatre salles au total ?
34. Combien de postes y a-t-il dans les salles Alpha et Beta réunies ?
35. Combien de postes y a-t-il dans les salles Gamma et Delta réunies ?
36. Quelle salle possède le plus de postes informatiques ?
37. Quelle salle possède le moins de postes informatiques ?
38. Combien de semaines faut-il pour suivre successivement Développement Web et Bases de Données ?
39. Combien de semaines faut-il pour suivre successivement Bases de Données et Administration Réseau ?
40. Quelle formation est la plus longue ?
41. Quelle formation est la plus courte ?
42. Si un étudiant obtient 80 % de réussite aux travaux pratiques et 70 % au projet final, quelle serait sa note globale selon les pondérations indiquées ?
43. Résume les horaires, les formations et les règles d'utilisation du matériel.

## D. Questions auxquelles le RAG ne doit pas inventer

44. Combien d'enseignants travaillent au Centre Horizon ?
45. Quel est le prix de la formation Développement Web ?
46. Qui est le directeur du centre ?
47. Quelle est l'adresse du Centre Horizon ?
48. Quel logiciel est installé sur les ordinateurs ?
49. Quel système d'exploitation utilisent les ordinateurs ?
50. Combien d'étudiants sont inscrits ?
51. Quelle est la date exacte de début des formations ?
52. Quel est le numéro de téléphone du centre ?
53. Le centre propose-t-il une formation en intelligence artificielle ?

Ces informations ne sont pas présentes dans le document.

---

# 4. Tests de séparation des documents

Ces questions permettent de vérifier que le RAG ne mélange pas les informations provenant de plusieurs fichiers.

## A. Document TXT vs Markdown

1. Quel est le prix d'un casque audio dans la bibliothèque ?
2. Quel est le nombre d'employés de NovaTech dans le service informatique ?
3. Quelle est l'adresse IP du réseau étudiant de NovaTech ?
4. Quel jour NovaTech effectue-t-elle ses inventaires ?
5. Combien d'ordinateurs possède la bibliothèque ?
6. Combien de jours les clients de NovaTech ont-ils pour retourner un produit ?

## B. Document Markdown vs PDF

7. Combien de salles possède NovaTech ?
8. Combien de postes possède le Centre Horizon ?
9. Quel est le prix moyen d'un clavier au Centre Horizon ?
10. Combien de semaines dure la formation Administration Réseau chez NovaTech ?
11. Quel est le seuil minimal de stock au Centre Horizon ?
12. À quelle heure commencent les cours du matin chez NovaTech ?

## C. TXT vs PDF

13. Quelle est l'adresse IP du serveur web du Centre Horizon ?
14. Combien de postes possède la salle Gamma de la bibliothèque ?
15. À quelle heure sont réalisées les sauvegardes du Centre Horizon ?
16. Combien de semaines dure la formation Développement Web dans la bibliothèque ?
17. Quel réseau est réservé aux étudiants du Centre Horizon ?
18. Combien d'ordinateurs possède la salle B du Centre Horizon ?

Pour toutes ces questions, le RAG doit reconnaître lorsque l'information demandée n'existe pas dans le document concerné.

---

# 5. Tests de synthèse

Ces questions sont utiles pour vérifier la capacité du RAG à produire une réponse plus complète.

1. Résume le fonctionnement général du RAG décrit dans le fichier TXT.
2. Résume la politique de sécurité de la bibliothèque.
3. Résume l'incident de sécurité du 15 juin.
4. Résume le fonctionnement de la gestion du stock de NovaTech.
5. Résume la politique de retour client de NovaTech.
6. Résume le fonctionnement du Centre Horizon.
7. Compare les trois formations du Centre Horizon.
8. Présente les règles principales de sécurité de la bibliothèque.
9. Présente les règles principales de gestion de NovaTech.
10. Présente les principales informations du Centre Horizon sous forme de tableau.

---

# 6. Test ultime : question hors contexte

Ces questions servent à vérifier le comportement du RAG lorsqu'il ne trouve aucune information pertinente.

1. Qui a créé le premier ordinateur ?
2. Quelle est la capitale de Madagascar ?
3. Comment fonctionne ChatGPT ?
4. Quel est le meilleur langage de programmation ?
5. Quelle est la météo aujourd'hui ?
6. Combien coûte un iPhone ?
7. Qui est le président actuel de Madagascar ?

## Résultat attendu

Si le RAG est configuré pour répondre exclusivement à partir des documents fournis, il devrait indiquer que ces informations ne sont pas disponibles dans les documents, au lieu de répondre à partir de ses connaissances générales.

---

# 7. Critères de validation du RAG

Pour chaque question, vérifier :

- **Pertinence** : la réponse correspond-elle réellement à la question ?
- **Exactitude** : les chiffres, noms, dates et adresses sont-ils corrects ?
- **Source** : le RAG utilise-t-il le bon document ?
- **Absence d'hallucination** : invente-t-il des informations absentes ?
- **Séparation** : mélange-t-il les informations entre TXT, MD et PDF ?
- **Questions complexes** : sait-il combiner plusieurs passages du même document ?
- **Questions hors contexte** : sait-il reconnaître qu'une information est absente ?
