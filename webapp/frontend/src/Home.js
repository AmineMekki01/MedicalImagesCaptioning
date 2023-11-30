import React from 'react';

function Home() {
  return (
    <div className="home">
      <header>
        <h1>Welcome to Our Website</h1>
      </header>
      <main>
        <section className="about">
          <h2>About Us</h2>
          <p>
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed vitae justo sit amet urna
            condimentum bibendum. Integer quis metus eu enim suscipit varius. Nulla facilisi.
            Phasellus nec massa non libero placerat hendrerit.
          </p>
          <p>
            Ut auctor semper libero, non fringilla sapien sagittis id. Suspendisse potenti. Donec
            euismod euismod lacus, non eleifend dolor facilisis ac.
          </p>
        </section>
      </main>
      <footer>
        <p>&copy; 2023 Our Website. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default Home;
