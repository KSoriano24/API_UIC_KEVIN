import crypto from 'crypto';

const KEY = Buffer.from(process.env.CRYPTO_KEY || 'g10wv0x$3cur3K3y#2024!AES256bit!!').slice(0, 32);

export function decryptText(encryptedBase64) {
    try {
        // Limpiar caracteres que HTTP puede haber codificado
        const clean = encryptedBase64
            .replace(/&#x2F;/g, '/')
            .replace(/&amp;/g, '&')
            .replace(/\s/g, '+');

        const combined = Buffer.from(clean, 'base64');
        const iv = combined.slice(0, 12);
        const authTag = combined.slice(combined.length - 16);
        const encrypted = combined.slice(12, combined.length - 16);

        const decipher = crypto.createDecipheriv('aes-256-gcm', KEY, iv);
        decipher.setAuthTag(authTag);
        return Buffer.concat([
            decipher.update(encrypted),
            decipher.final()
        ]).toString('utf8');

    } catch (e) {
        console.error('Error decrypt:', e.message);
        throw new Error('Dato cifrado inválido o manipulado');
    }
}

export function decryptFile(encryptedBuffer) {
    try {
        const iv = encryptedBuffer.slice(0, 12);
        const authTag = encryptedBuffer.slice(encryptedBuffer.length - 16);
        const encrypted = encryptedBuffer.slice(12, encryptedBuffer.length - 16);

        const decipher = crypto.createDecipheriv('aes-256-gcm', KEY, iv);
        decipher.setAuthTag(authTag);
        return Buffer.concat([decipher.update(encrypted), decipher.final()]);
    } catch {
        throw new Error('Archivo inválido o manipulado');
    }
}