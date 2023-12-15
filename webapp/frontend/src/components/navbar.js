import React, { useState } from 'react';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemText from '@mui/material/ListItemText';
import Divider from '@mui/material/Divider';
import Toolbar from '@mui/material/Toolbar';
import AppBar from '@mui/material/AppBar';
import CssBaseline from '@mui/material/CssBaseline';
import IconButton from '@mui/material/IconButton';
import MenuIcon from '@mui/icons-material/Menu';
import Drawer from '@mui/material/Drawer';
import Button from '@mui/material/Button';
import { useNavigate } from 'react-router-dom';
import { styled } from '@mui/material/styles';




const drawerWidth = 240;
const navItems = ['Generate' , 'About' ];


const CustomList = styled('ul')({
    listStyle: 'none',
    paddingLeft: 0,
});

const CustomBullet = styled('span')({
    display: 'inline-block',
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    backgroundColor: '#1976d2',
    marginRight: '8px',
});

function DrawerAppBar(props) {
    const { window } = props;
    const [mobileOpen, setMobileOpen] = useState(false);
    const [selectedItem, setSelectedItem] = useState('Home');
    const navigate = useNavigate();

    const handleDrawerToggle = () => {
        setMobileOpen((prevState) => !prevState);
    };

    const handleItemClick = (item) => {
        /*setSelectedItem(item);*/
        navigate('/'+item);
        setMobileOpen(false); 
    };

    


    const drawer = (
        <Box onClick={handleDrawerToggle} sx={{ textAlign: 'center' }}>
            <Typography variant="h6" sx={{ my: 2 }}>
                <img src={process.env.PUBLIC_URL + '/images/logo_kaiy.png'} alt="Logo" style={{ width: '70px', height: '70px', marginRight: '10px' }} />
            </Typography>
            <Divider />
            <List>
                {navItems.map((item) => (
                    <ListItem key={item} disablePadding>
                        <ListItemButton
                            sx={{ textAlign: 'center' }}
                            selected={selectedItem === item}
                            onClick={() => handleItemClick(item)}
                        >
                            <ListItemText primary={item} />
                        </ListItemButton>
                    </ListItem>
                ))}
            </List>
        </Box>
    );

    const container = window !== undefined ? () => window().document.body : undefined;

    return (
        <Box sx={{ display: 'flex' }}>
            <CssBaseline />
            <AppBar component="nav">
                <Toolbar>
                    <IconButton
                        color="inherit"
                        aria-label="open drawer"
                        edge="start"
                        onClick={handleDrawerToggle}
                        sx={{ mr: 2, display: { sm: 'none' } }}
                    >
                        <MenuIcon />
                    </IconButton>
                    <Typography
                        variant="h6"
                        component="div"
                        sx={{ flexGrow: 1, display: { xs: 'none', sm: 'block' } }}
                    >
                        Virtuosos
                    </Typography>
                    <Box sx={{ display: { xs: 'none', sm: 'block' } }}>
                        {navItems.map((item) => (
                            <Button
                                key={item}
                                sx={{ color: '#fff' }}
                                onClick={() => handleItemClick(item)}
                                variant={selectedItem === item ? 'contained' : 'text'}
                            >
                                {item}
                            </Button>
                        ))}
                    </Box>
                </Toolbar>
            </AppBar>
            <nav>
                <Drawer
                    container={container}
                    variant="temporary"
                    open={mobileOpen}
                    onClose={handleDrawerToggle}
                    ModalProps={{
                        keepMounted: true, 
                    }}
                    sx={{
                        display: { xs: 'block', sm: 'none' },
                        '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
                    }}
                >
                    {drawer}
                </Drawer>
            </nav>
            <Box component="main" sx={{ p: 3 }}>
                <Toolbar />
                {/*getContent()*/}
            </Box>
        </Box>
    );
}

export default DrawerAppBar;
