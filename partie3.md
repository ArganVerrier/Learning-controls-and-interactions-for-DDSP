
## VAE for fixed-length samples

*divisez les taches et faites des groupes de travail*

**préparation des données et extraction des enveloppes de descripteurs (f0,loudness)**

élève(s):


* SOL ordinario == notes individuelles (+ labels)
* couper les samples à une durée fixe (e.g. 3 secondes)
* extraire 2 enveloppes à 100Hz (ou 250Hz) par sample (e.g. 3sec * 100Hz * 2 = 600 points)

**adaptation d'un VAE à la reconstruction de paires d'enveloppes**

élève(s):

* tester un modèle seulement avec des couches linéaires
* tester un modèle convolution + linéaire

**visualisation et évaluation**

élève(s):

* faire différents codes qui permettent de visualiser les paires d'enveloppes, la modulation autour du pitch de la note etc.
* faire le code de synthèse avec le DDSP pré-entrainé
