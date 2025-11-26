/**
 * Classe Perimetre - Calcul de périmètre ((a + b) * c)
 * TODO: EXERCICE 3 - Implémenter la méthode perim()
 */
public class Perimetre {

    /**
     * TODO: Calculer le périmètre
     *
     * Formule: (a + b) * c
     * Utiliser Addition.add() et Produit.mult()
     *
     * @param a Premier côté
     * @param b Deuxième côté
     * @param c Facteur
     * @return (a + b) * c
     */
    public static int perim(int a, int b, int c) {
        // TODO: Calculer somme avec Addition.add(a, b)
        // TODO: Multiplier par c avec Produit.mult(somme, c)
        return Produit.mult(Addition.add(a,b),c);
    }
}
