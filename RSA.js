class RSA {
    constructor() {
        this.p = 61; 
        this.q = 53;
        this.n = this.p * this.q; //
        this.phi = (this.p - 1) * (this.q - 1); // Euler's totient function
        this.e = 17; // e is chosen such that 1 < e < phi and gcd(e, phi) = 1
        this.d = this.modInverse(this.e, this.phi); 
    }

    modInverse(e, phi) {
        let m0 = phi, t, q;
        let x0 = 0, x1 = 1;
        if (phi == 1) return 0;
        while (e > 1) {
            q = Math.floor(e / phi);
            t = phi;
            phi = e % phi;
            e = t;
            t = x0;
            x0 = x1 - q * x0;
            x1 = t;
        }
        if (x1 < 0) x1 += m0;
        return x1;
    }

    encryptMessage(message) {
        let encrypted = [];
                for (let i = 0; i < message.length; i++) {
                    let m = message.charCodeAt(i);
                    let c = BigInt(m) ** BigInt(this.e) % BigInt(this.n);
                    encrypted.push(c.toString());
                }
                return encrypted.join(" ");
    }

    decryptMessage(encryptedMessage) {
        let decrypted = "";
        let parts = encryptedMessage.split(" ");
        for (let part of parts) {
            let c = BigInt(part);
            let m = c ** BigInt(this.d) % BigInt(this.n);
            decrypted += String.fromCharCode(Number(m));
        }
        return decrypted;
    }
}

let rsa = new RSA();

function encryptMessage() {
    let message = document.getElementById("message").value;
    let encryptedText = rsa.encryptMessage(message);
    document.getElementById("encryptedText").innerText = encryptedText;
}

function decryptMessage() {
    let encryptedText = document.getElementById("encryptedText").innerText;
    let decryptedText = rsa.decryptMessage(encryptedText);
    document.getElementById("decryptedText").innerText = decryptedText;
}