const { addMessage, getMessages, processData } = require("../controllers/messageController");
const authenticateToken = require('../middleware/tokenAuthentication');
const router = require("express").Router();

router.post("/addmsg/", authenticateToken, addMessage);
router.post("/getmsg/", authenticateToken, getMessages);
router.post("/processdata/", authenticateToken, processData);

module.exports = router;
