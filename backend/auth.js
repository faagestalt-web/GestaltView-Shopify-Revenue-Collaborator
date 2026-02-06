/*
 * auth.js – Shopify OAuth placeholder
 *
 * This module provides stub middleware for verifying an authenticated
 * Shopify session.  In a production application you would use the
 * @shopify/shopify-api package to verify request HMAC signatures,
 * exchange authorization codes for access tokens, and check that
 * requests originate from the Shopify Admin.  For this example we
 * simply allow all requests through.
 */

// Placeholder verification middleware
function verifyAuth(req, res, next) {
  // TODO: validate the session token or HMAC signature
  // You might parse req.headers.authorization or cookies here
  next();
}

module.exports = { verifyAuth };
