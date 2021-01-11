#1 On ne doit pas utiliser une relu sur mu, il peut être positif et négatif (comme les valeurs d'une gaussienne centrée en zéro) :

Ex:
mu = F.relu(self.fc1(encoder)) -> Faux
mu = self.fc1(encoder) -> Correct

#2 La sortie d'encoder que on appelle sigma est en fait une log variance, c'est pour ça que'on fait sigma.exp() d'ailleurs c'est bien car c'est plus stable que directement output la variance par contre une log variance peut être négative c'est contraire à :

Ex:
sigma = F.softplus(self.fc2(encoder)) -> Faux
sigma = self.fc2(encoder) -> Correct


#3 Si les résultats s'amériorent pas, on peut tracer aussi la courbe des loss au cours de l'entrainement (eg utilises tensorboard, ça marche avec pytorch). Si on voit que la loss de kl_div descend très bas mais que la loss de reconstruction non c'est que la kl_div est "trop forte", on peut ajouter un parametre "beta_vae" (un float) qui ajuste cette force comme :
    
full_loss = loss + kl_div * beta_vae avec e.g. beta_vae=0.1 -> marche mieux et il faut aussi regler beta_vae avec différentes valeurs


#4 Pour un réseau à la taille de vecteur d'activation comme self.flat = nn.Flatten() # 12877 -> 6272, le gradient peut faire un peu nimp, un layer que il nous conseille d'utiliser la batchnorm, qui permet d'ajuster la variance des tes activations et faire un gradient plus  homogène, il y a plein d'activations, pour l'instant la batchnorm en général ça devrait que vous faciliter la tache (pour des problèmes plus compliqué, c'est pas toujours ça à utiliser !)

En règle général on peut faire les couches cachées comme layer + normalisation + activation :

Ex :
self.conv12 = nn.Sequential(nn.Conv2d(nin, 64, kernel_size=4, stride=2, padding=1),nn.BatchNorm2d(64))
z = F.relu(self.conv12(z))

Dans le doc, BatchNorm2d prend la taille de channel output du layer d'avant et pour les Linear, on peut ajouter BatchNorm1d, on peut mettre dans un Sequential là car les normalizations ne changent pas la dimension ! donc les print que tu as fais pour t'assurer de la taille de sortie de chaque couche (c'est bien ça !) restent valables

On peut ajouter la normalization aux layer convolution et linéraire (mais pas ceux d'output pour mu et sigma ! mais pas celui d'output pour la reconstruction !)

L'idée, c'est de normaliser les activations cachées, les output en général on a just à choisir une activation qui correspond au range de ce que on veut générer et une loss qui correspond.

#5 Pour la loss, il est plus interessant d'utiliser "reduction='mean'", ça nous permettra de comparer différents différentes évolution de loss, pour différents jeu de données


Tuto pour choisir les hyperparamètres: 
https://towardsdatascience.com/guide-to-choosing-hyperparameters-for-your-neural-networks-38244e87dafe
