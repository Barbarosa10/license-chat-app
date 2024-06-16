const { processData } = require("../controllers/messageController");
const authenticateToken = require('../middleware/tokenAuthentication');
const router = require("express").Router();

router.post("/processdata/", authenticateToken, processData);

module.exports = router;