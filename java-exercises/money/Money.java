/**
 * Classe Money - Gestion de monnaie et devise
 * Exercice II - Workshop 1
 */
public class Money {

    private int mAmount;
    private String mCurrency;

    /**
     * Constructeur (DÉJÀ IMPLÉMENTÉ)
     */
    public Money(int amount, String currency) {
        this.mAmount = amount;
        this.mCurrency = currency;
    }

    public int amount() {
        return mAmount;
    }

    public String currency() {
        return mCurrency;
    }

    /**
     * TODO: EXERCICE 1 - Implémenter l'addition de deux monnaies
     *
     * L'addition se fait seulement si les devises sont identiques.
     * Si les devises sont différentes, lever une Exception.
     *
     * @param m La monnaie à ajouter
     * @return Une nouvelle Money avec la somme
     * @throws Exception si les devises sont différentes
     */
    public Money add(Money m) throws Exception {
        // TODO: Vérifier si this.currency().equals(m.currency())
        // TODO: Si oui, retourner new Money(this.amount() + m.amount(), this.currency())
        // TODO: Si non, throw new Exception("Not Same currency")
        if (this.currency().equals(m.currency())) {
        return new Money(this.amount() + m.amount(), this.currency());
        }
        throw new Exception("Not Same currency");
        
    }
}
