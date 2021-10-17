import pygame
from settings import Settings
from game_stats import GameStats
from button import Button
from scoreboard import Scoreboard
from ship import Ship
from pygame.sprite import Group
import sys
import numpy as np
import pygame
from bullet import Bullet
from alien import Alien

pygame.init()

class SpaceInvadersAI:
    def __init__(self):
        self.ai_settings = Settings()
        self.screen = pygame.display.set_mode((self.ai_settings.screen_width, self.ai_settings.screen_height))
        pygame.display.set_caption("Alien Invasion")
        self.stats = GameStats(self.ai_settings)
        self.sb = Scoreboard(self.ai_settings, self.screen, self.stats)
        self.ship = Ship(self.ai_settings, self.screen)
        self.aliens = Group()
        self.bullets = Group()
        self.play_button = Button(self.ai_settings, self.screen, 'ye')
        self.alien = Alien(self.ai_settings, self.screen)
        self.reset()

    def reset(self):
        self.ai_settings.initialize_dynamic_settings()
        self.stats.reset_stats()
        self.sb.prep_score()
        self.sb.prep_high_score()
        self.sb.prep_level()
        self.sb.prep_ships()
        self.aliens.empty()
        self.bullets.empty()
        self.create_fleet()
        self.ship.center_ship()

    def check_events(self, action, draw_screen):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        if np.array_equal(action, [1, 0]):
            self.ship.moving_left = False
            self.ship.moving_right = True

        elif np.array_equal(action, [0, 1]):
            self.ship.moving_left = True
            self.ship.moving_right = False

        reward = 0
        game_over = False

        self.ship.update()
        H_R = self.update_bullets()
        self.fire_bullet()
        Score = self.stats.score
        L = self.update_aliens()
        if draw_screen:
            self.update_screen()

        if L == True:
            reward = -10
            game_over = True
            return reward, game_over, Score

        if H_R == 'Round':
            return reward, game_over, self.stats.score

        if H_R == 'Hit':
            reward = 2
            return reward, game_over, self.stats.score

        return reward, game_over, self.stats.score

    def check_high_score(self):
        if self.stats.score > self.stats.high_score:
            self.stats.high_score = self.stats.score
            self.sb.prep_high_score()

    def update_bullets(self):
        self.bullets.update()
        for bullet in self.bullets.copy():
            if bullet.rect.bottom <= 0:
                self.bullets.remove(bullet)

        H_R = self.check_bullet_alien_collisions()
        return H_R

    def check_bullet_alien_collisions(self):
        collisions = pygame.sprite.groupcollide(self.bullets, self.aliens, True, True)
        col = False
        if collisions:
            for aliens in collisions.values():
                self.stats.score += self.ai_settings.alien_points
                self.sb.prep_score()
            self.check_high_score()
            col = 'Hit'

        if len(self.aliens) == 0:
            col = 'Round'
            self.bullets.empty()
            self.ai_settings.increase_speed()

            self.stats.level += 1
            self.sb.prep_level()
            self.create_fleet()
        return col

    def fire_bullet(self):
        if len(self.bullets) < self.ai_settings.bullets_allowed:
            new_bullet = Bullet(self.ai_settings, self.screen, self.ship)
            self.bullets.add(new_bullet)

    def get_number_rows(self, alien_height):
        available_space_y = (self.ai_settings.screen_height - (3 * alien_height) - self.ship.rect.height) #ship.rect.height, alien.rect.height
        number_rows = int(available_space_y / (2 * alien_height))
        return number_rows

    def get_number_aliens_x(self, alien_width):
        available_space_x = self.ai_settings.screen_width - 2 * alien_width
        number_aliens_x = int(available_space_x / (2* alien_width))
        return number_aliens_x

    def create_alien(self, alien_number, row_number):
        alien = Alien(self.ai_settings, self.screen)
        alien_width = alien.rect.width
        alien.x = alien_width + 2 * alien_width * alien_number
        alien.rect.x = alien.x
        alien.rect.y = alien.rect.height + 2 * alien.rect.height * row_number
        self.aliens.add(alien)

    def create_fleet(self):
        number_aliens_x = self.get_number_aliens_x(self.alien.rect.width)
        number_rows = self.get_number_rows(self.alien.rect.height)

        for row_number in range(number_rows):
            for alien_number in range(number_aliens_x):
                self.create_alien(alien_number, row_number)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    def check_fleet_edges(self):
        for alien in self.aliens.sprites():
            if alien.check_edges():
                self.change_fleet_direction()
                break

    def change_fleet_direction(self):
        for alien in self.aliens.sprites():
            alien.rect.y += self.ai_settings.fleet_drop_speed
        self.ai_settings.fleet_direction *= -1

    def ship_hit(self):
        if self.stats.ships_left > 0:
            self.stats.ships_left -= 1
            self.sb.prep_ships()

        else:
            self.reset()

        self.aliens.empty()
        self.bullets.empty()

        self.create_fleet()
        self.ship.center_ship()

    def check_aliens_bottom(self):
        screen_rect = self.screen.get_rect()
        for alien in self.aliens.sprites():
            if alien.rect.bottom >= screen_rect.bottom:
                self.ship_hit()
                return True

        return False

    def update_aliens(self):
        self.check_fleet_edges()
        self.aliens.update()
        if pygame.sprite.spritecollideany(self.ship, self.aliens):
            self.ship_hit()
            return True
        return self.check_aliens_bottom()

    def update_screen(self):
        self.screen.fill(self.ai_settings.bg_color)
        for bullet in self.bullets.sprites():
            bullet.draw_bullet()
        self.ship.blitme()
        self.aliens.draw(self.screen)
        self.sb.show_score()
        pygame.display.flip()
        